"""
TabNet model for fraud detection.
Implements attention-based intrinsic interpretability.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TabNetFraudDetector:
    """
    TabNet-based fraud detector with built-in interpretability.
    TabNet provides feature importance through sparse attention masks.
    """
    
    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        lambda_sparse: float = 1e-4,
        optimizer_fn: str = 'adam',
        optimizer_params: Optional[Dict] = None,
        scheduler_params: Optional[Dict] = None,
        mask_type: str = 'sparsemax',
        verbose: int = 1,
        device_name: str = 'auto',
        seed: int = 42
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.mask_type = mask_type
        self.verbose = verbose
        self.device_name = device_name
        self.seed = seed
        
        self.optimizer_params = optimizer_params or {'lr': 2e-2, 'weight_decay': 1e-5}
        self.scheduler_params = scheduler_params or {
            "step_size": 10,
            "gamma": 0.9
        }
        
        self.model: Optional[TabNetClassifier] = None
        self.feature_names: List[str] = []
        self.cat_idxs: List[int] = []
        self.cat_dims: List[int] = []
        
    def build_model(self, cat_idxs: Optional[List[int]] = None, cat_dims: Optional[List[int]] = None):
        """Build TabNet classifier."""
        
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        
        self.model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=self.optimizer_params,
            scheduler_params=self.scheduler_params,
            mask_type=self.mask_type,
            verbose=self.verbose,
            device_name=self.device_name,
            seed=self.seed
        )
        
        logger.info(f"Built TabNet model with {self.n_steps} steps")
        return self
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_epochs: int = 100,
        batch_size: int = 1024,
        patience: int = 15,
        weights: int = 1,
        eval_metric: List[str] = ['auc']
    ) -> Dict[str, Any]:
        """Train TabNet model."""
        
        logger.info("Training TabNet model...")
        
        fit_params = {
            'X_train': X_train,
            'y_train': y_train,
            'eval_set': [(X_val, y_val)] if X_val is not None else None,
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'patience': patience,
            'weights': weights,
            'eval_metric': eval_metric
        }
        
        # Remove None values
        fit_params = {k: v for k, v in fit_params.items() if v is not None}
        
        self.model.fit(**fit_params)
        
        # Get training history
        history = self.model.history
        
        metrics = {}
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val, prefix='val_')
        
        logger.info(f"TabNet training completed")
        logger.info(f"Best epoch: {self.model.best_epoch}")
        
        return {**metrics, 'history': history}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        prefix: str = ''
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        
        metrics = {
            f'{prefix}roc_auc': roc_auc_score(y, y_prob),
            f'{prefix}average_precision': average_precision_score(y, y_prob),
            f'{prefix}f1': f1_score(y, y_pred),
            f'{prefix}precision': precision_score(y, y_pred, zero_division=0),
            f'{prefix}recall': recall_score(y, y_pred, zero_division=0),
        }
        
        return metrics
    
    def explain(
        self, 
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get TabNet explanations via attention masks.
        
        Returns:
            Dictionary with 'masks' (attention weights per step) and 
            'aggregate' (aggregated feature importance)
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Get masks from TabNet
        masks = self.model.explain(X)
        
        # Aggregate masks across steps (weighted by importance)
        # masks is a list of arrays, one per step
        if isinstance(masks, tuple):
            masks_list = list(masks)
        else:
            masks_list = [masks]
        
        # Aggregate across steps
        aggregate_mask = np.zeros_like(masks_list[0])
        for i, mask in enumerate(masks_list):
            weight = (i + 1) / len(masks_list)  # Later steps get more weight
            aggregate_mask += mask * weight
        
        aggregate_mask /= sum(range(1, len(masks_list) + 1))
        
        result = {
            'masks': masks_list,
            'aggregate': aggregate_mask,
            'mean_importance': aggregate_mask.mean(axis=0)
        }
        
        if feature_names:
            result['feature_importance'] = pd.DataFrame({
                'feature': feature_names,
                'importance': result['mean_importance']
            }).sort_values('importance', ascending=False)
        
        return result
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance from TabNet."""
        # TabNet doesn't provide global importance directly
        # We use the reducer weights as proxy
        importance = np.zeros(len(self.feature_names))
        
        if hasattr(self.model, 'network'):
            # Extract importance from network parameters
            for name, param in self.model.network.named_parameters():
                if 'feat_transformers' in name and 'weight' in name:
                    importance += param.abs().mean(dim=0).detach().cpu().numpy()
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """Save TabNet model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save using TabNet's save method
        self.model.save_model(path)
        
        # Save additional metadata
        metadata = {
            'feature_names': self.feature_names,
            'cat_idxs': self.cat_idxs,
            'cat_dims': self.cat_dims,
            'params': {
                'n_d': self.n_d,
                'n_a': self.n_a,
                'n_steps': self.n_steps,
                'gamma': self.gamma,
                'lambda_sparse': self.lambda_sparse
            }
        }
        joblib.dump(metadata, f"{path}_metadata.joblib")
        
        logger.info(f"Saved TabNet model to {path}")
    
    def load(self, path: str):
        """Load TabNet model."""
        self.model = TabNetClassifier()
        self.model.load_model(path)
        
        # Load metadata
        metadata = joblib.load(f"{path}_metadata.joblib")
        self.feature_names = metadata['feature_names']
        self.cat_idxs = metadata['cat_idxs']
        self.cat_dims = metadata['cat_dims']
        
        logger.info(f"Loaded TabNet model from {path}")
        return self


class TabNetSurrogate:
    """
    Lightweight TabNet surrogate for fast inference.
    Can be used as a faster alternative to the full TabNet model.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.model = None
        
    def build(self) -> nn.Module:
        """Build a simple feedforward surrogate."""
        
        class SurrogateNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim // 2, output_dim),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.net(x)
        
        self.model = SurrogateNN(self.input_dim, self.hidden_dim, self.output_dim)
        return self.model
    
    def distill(
        self,
        teacher_model: TabNetFraudDetector,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 512,
        lr: float = 1e-3
    ):
        """Distill knowledge from TabNet teacher to surrogate."""
        
        if self.model is None:
            self.build()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        # Get teacher predictions
        teacher_probs = teacher_model.predict_proba(X_train)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_teacher = torch.FloatTensor(teacher_probs).unsqueeze(1).to(device)
        y_true = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        
        # Training loop
        self.model.train()
        n_samples = len(X_train)
        
        for epoch in range(epochs):
            total_loss = 0
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X_tensor[batch_idx]
                y_teacher_batch = y_teacher[batch_idx]
                y_true_batch = y_true[batch_idx]
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                
                # Distillation loss: combination of teacher and true labels
                loss = 0.7 * criterion(outputs, y_teacher_batch) + \
                       0.3 * criterion(outputs, y_true_batch)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Distillation Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_samples:.4f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict using surrogate model."""
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy().squeeze()
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        return (self.predict_proba(X) >= threshold).astype(int)
