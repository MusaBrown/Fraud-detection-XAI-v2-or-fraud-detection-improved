"""
FastSHAP: Neural Network-based SHAP Approximation
Implements a surrogate model to approximate SHAP values efficiently.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastSHAPSurrogate(nn.Module):
    """
    Neural network surrogate model for FastSHAP.
    Learns to predict SHAP values directly from input features.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        super().__init__()
        self.input_dim = input_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer: predicts SHAP value for each feature
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Learned baseline (expected value)
        self.baseline = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting SHAP values.
        
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            SHAP values (batch_size, input_dim)
        """
        return self.network(x)
    
    def predict_with_baseline(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict SHAP values and baseline.
        
        Returns:
            Tuple of (shap_values, baseline)
        """
        shap_values = self.forward(x)
        return shap_values, self.baseline


class SHAPValueDataset(Dataset):
    """Dataset for training FastSHAP on pre-computed SHAP values."""
    
    def __init__(
        self,
        X: np.ndarray,
        shap_values: np.ndarray,
        expected_values: Optional[np.ndarray] = None
    ):
        self.X = torch.FloatTensor(X)
        self.shap_values = torch.FloatTensor(shap_values)
        self.expected_values = torch.FloatTensor(expected_values) if expected_values is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        item = {
            'X': self.X[idx],
            'shap_values': self.shap_values[idx]
        }
        if self.expected_values is not None:
            item['expected_value'] = self.expected_values[idx]
        return item


class FastSHAPExplainer:
    """
    FastSHAP explainer using neural network approximation.
    Trains a surrogate model to predict SHAP values efficiently.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 100,
        patience: int = 10,
        device: Optional[str] = None
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        self.surrogate: Optional[FastSHAPSurrogate] = None
        self.is_fitted = False
        self.training_history: List[Dict] = []
        
        self.feature_names: Optional[List[str]] = None
        self.expected_value: Optional[float] = None
        
    def build_model(self):
        """Build the surrogate model."""
        self.surrogate = FastSHAPSurrogate(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        logger.info(f"Built FastSHAP surrogate with {sum(p.numel() for p in self.surrogate.parameters())} parameters")
        return self
    
    def fit(
        self,
        X_train: np.ndarray,
        shap_values_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        shap_values_val: Optional[np.ndarray] = None,
        expected_value: Optional[float] = None
    ) -> Dict:
        """
        Train FastSHAP surrogate on pre-computed SHAP values.
        
        Args:
            X_train: Training features
            shap_values_train: Exact SHAP values for training
            X_val: Validation features
            shap_values_val: Exact SHAP values for validation
            expected_value: Expected value (baseline) from SHAP
            
        Returns:
            Training history
        """
        if self.surrogate is None:
            self.build_model()
        
        self.expected_value = expected_value
        
        # Create datasets
        train_dataset = SHAPValueDataset(X_train, shap_values_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = None
        if X_val is not None and shap_values_val is not None:
            val_dataset = SHAPValueDataset(X_val, shap_values_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        # Training setup
        optimizer = torch.optim.Adam(
            self.surrogate.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.patience // 2
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("Training FastSHAP surrogate...")
        
        for epoch in range(self.epochs):
            # Training
            self.surrogate.train()
            train_loss = 0.0
            
            for batch in train_loader:
                X_batch = batch['X'].to(self.device)
                shap_batch = batch['shap_values'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                pred_shap = self.surrogate(X_batch)
                
                # Loss: MSE between predicted and true SHAP values
                loss = F.mse_loss(pred_shap, shap_batch)
                
                # Add L2 regularization
                l2_reg = 0.0
                for param in self.surrogate.parameters():
                    l2_reg += torch.norm(param, p=2)
                loss += 1e-5 * l2_reg
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(X_batch)
            
            train_loss /= len(train_dataset)
            
            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            self.training_history.append(epoch_result)
            
            if (epoch + 1) % 10 == 0:
                log_msg = f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.6f}"
                logger.info(log_msg)
        
        self.is_fitted = True
        
        # Compute fidelity metrics on validation set
        fidelity_metrics = {}
        if X_val is not None and shap_values_val is not None:
            fidelity_metrics = self.compute_fidelity(X_val, shap_values_val)
            logger.info(f"Validation fidelity - Pearson: {fidelity_metrics.get('pearson', 'N/A'):.4f}")
        
        return {
            'history': self.training_history,
            'fidelity': fidelity_metrics
        }
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate model on validation set."""
        self.surrogate.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                X_batch = batch['X'].to(self.device)
                shap_batch = batch['shap_values'].to(self.device)
                
                pred_shap = self.surrogate(X_batch)
                loss = F.mse_loss(pred_shap, shap_batch)
                
                val_loss += loss.item() * len(X_batch)
        
        val_loss /= len(val_loader.dataset)
        return val_loss
    
    def explain(self, X: np.ndarray) -> Dict:
        """
        Generate FastSHAP explanations.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with SHAP values and metadata
        """
        if not self.is_fitted:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        self.surrogate.eval()
        
        start_time = time.time()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            shap_values = self.surrogate(X_tensor).cpu().numpy()
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'shap_values': shap_values,
            'expected_value': self.expected_value,
            'latency_ms': latency_ms,
            'feature_names': self.feature_names
        }
    
    def explain_single(self, x: np.ndarray) -> Dict:
        """Explain a single instance (optimized for real-time)."""
        return self.explain(x.reshape(1, -1))
    
    def compute_fidelity(
        self,
        X: np.ndarray,
        true_shap_values: np.ndarray,
        top_k: int = 10
    ) -> Dict:
        """
        Compute fidelity metrics comparing FastSHAP to exact SHAP.
        
        Args:
            X: Input features
            true_shap_values: Ground truth SHAP values
            top_k: Number of top features for rank correlation
            
        Returns:
            Dictionary of fidelity metrics
        """
        pred = self.explain(X)
        pred_shap = pred['shap_values']
        
        # Pearson correlation
        from scipy.stats import pearsonr, spearmanr
        
        # Flatten for overall correlation
        flat_true = true_shap_values.flatten()
        flat_pred = pred_shap.flatten()
        
        pearson_r, _ = pearsonr(flat_true, flat_pred)
        
        # Per-instance metrics
        pearson_per_instance = []
        spearman_topk = []
        
        for i in range(len(X)):
            # Pearson per instance
            if np.std(true_shap_values[i]) > 0 and np.std(pred_shap[i]) > 0:
                r, _ = pearsonr(true_shap_values[i], pred_shap[i])
                pearson_per_instance.append(r)
            
            # Rank correlation for top-k features
            top_true_idx = np.argsort(np.abs(true_shap_values[i]))[-top_k:]
            top_pred_idx = np.argsort(np.abs(pred_shap[i]))[-top_k:]
            
            # Create rank vectors
            true_ranks = np.zeros(len(X[0]))
            pred_ranks = np.zeros(len(X[0]))
            true_ranks[top_true_idx] = 1
            pred_ranks[top_pred_idx] = 1
            
            if np.std(true_ranks) > 0 and np.std(pred_ranks) > 0:
                rho, _ = spearmanr(true_ranks, pred_ranks)
                spearman_topk.append(rho)
        
        # MSE
        mse = np.mean((true_shap_values - pred_shap) ** 2)
        mae = np.mean(np.abs(true_shap_values - pred_shap))
        
        return {
            'pearson': pearson_r,
            'pearson_per_instance_mean': np.mean(pearson_per_instance),
            'pearson_per_instance_std': np.std(pearson_per_instance),
            'spearman_topk_mean': np.mean(spearman_topk),
            'spearman_topk_std': np.std(spearman_topk),
            'mse': mse,
            'mae': mae,
            'mean_latency_ms': pred['latency_ms'] / len(X)
        }
    
    def save(self, path: str):
        """Save FastSHAP model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.surrogate.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'expected_value': self.expected_value,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, path)
        
        logger.info(f"Saved FastSHAP model to {path}")
    
    def load(self, path: str):
        """Load FastSHAP model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.input_dim = checkpoint['input_dim']
        self.hidden_dims = checkpoint['hidden_dims']
        self.dropout = checkpoint['dropout']
        self.expected_value = checkpoint['expected_value']
        self.feature_names = checkpoint['feature_names']
        self.is_fitted = checkpoint['is_fitted']
        
        self.build_model()
        self.surrogate.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded FastSHAP model from {path}")
        return self


class FastSHAPTrainer:
    """
    Helper class to train FastSHAP from a trained model.
    Generates training data using exact TreeSHAP.
    """
    
    def __init__(
        self,
        model,
        fastshap_config: Optional[Dict] = None
    ):
        self.model = model
        self.config = fastshap_config or {}
        
    def train_from_model(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        tree_explainer,
        max_training_samples: int = 10000
    ) -> FastSHAPExplainer:
        """
        Train FastSHAP using exact TreeSHAP as teacher.
        
        Args:
            X_train: Training features
            X_val: Validation features
            tree_explainer: Fitted TreeSHAP explainer
            max_training_samples: Max samples for training
            
        Returns:
            Trained FastSHAP explainer
        """
        logger.info("Generating SHAP values for FastSHAP training...")
        
        # Subsample if needed
        if len(X_train) > max_training_samples:
            indices = np.random.choice(len(X_train), max_training_samples, replace=False)
            X_train_sample = X_train[indices]
        else:
            X_train_sample = X_train
        
        # Generate exact SHAP values
        logger.info(f"Computing exact SHAP for {len(X_train_sample)} training samples...")
        train_result = tree_explainer.explain(X_train_sample)
        shap_train = train_result['shap_values']
        expected_value = train_result['expected_value']
        
        logger.info(f"Computing exact SHAP for {len(X_val)} validation samples...")
        val_result = tree_explainer.explain(X_val[:min(len(X_val), 2000)])
        shap_val = val_result['shap_values']
        
        # Initialize and train FastSHAP
        input_dim = X_train.shape[1]
        fastshap = FastSHAPExplainer(
            input_dim=input_dim,
            **self.config
        )
        
        logger.info("Training FastSHAP surrogate...")
        fastshap.fit(
            X_train=X_train_sample,
            shap_values_train=shap_train,
            X_val=X_val[:min(len(X_val), 2000)],
            shap_values_val=shap_val,
            expected_value=expected_value
        )
        
        return fastshap


def create_fastshap_from_model(
    model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    tree_explainer,
    hidden_dims: List[int] = [256, 128, 64],
    **kwargs
) -> FastSHAPExplainer:
    """
    Convenience function to create and train FastSHAP.
    
    Args:
        model: Trained model
        X_train: Training data
        X_val: Validation data
        tree_explainer: TreeSHAP explainer
        hidden_dims: Hidden layer dimensions
        **kwargs: Additional FastSHAP parameters
        
    Returns:
        Trained FastSHAP explainer
    """
    config = {
        'hidden_dims': hidden_dims,
        **kwargs
    }
    
    trainer = FastSHAPTrainer(model, config)
    return trainer.train_from_model(X_train, X_val, tree_explainer)
