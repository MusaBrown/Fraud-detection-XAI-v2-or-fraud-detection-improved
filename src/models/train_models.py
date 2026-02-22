"""
Model training module for fraud detection.
Implements XGBoost, LightGBM, and CatBoost with hyperparameter optimization.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, 
    recall_score, average_precision_score, classification_report
)
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudModelTrainer:
    """
    Unified trainer for tree-based fraud detection models.
    Supports XGBoost, LightGBM, and CatBoost.
    """
    
    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model = None
        self.feature_names: List[str] = []
        self.training_time: float = 0.0
        
    def build_model(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        class_weight: Optional[str] = 'balanced',
        **kwargs
    ):
        """Build model with specified hyperparameters."""
        
        if self.model_type == 'xgboost':
            import xgboost as xgb
            
            scale_pos_weight = kwargs.get('scale_pos_weight', 10)
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale_pos_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                tree_method='hist',
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='auc',
                use_label_encoder=False
            )
            
        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            
            is_unbalance = class_weight == 'balanced'
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=2**max_depth - 1,
                class_weight=class_weight if not is_unbalance else None,
                is_unbalance=is_unbalance,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )
            
        elif self.model_type == 'catboost':
            from catboost import CatBoostClassifier
            
            self.model = CatBoostClassifier(
                iterations=n_estimators,
                depth=max_depth,
                learning_rate=learning_rate,
                auto_class_weights='Balanced' if class_weight == 'balanced' else None,
                l2_leaf_reg=3.0,
                random_seed=self.random_state,
                verbose=False,
                thread_count=-1,
                loss_function='Logloss',
                eval_metric='AUC'
            )
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Built {self.model_type} model")
        return self
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50,
        categorical_features: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Train the model with early stopping."""
        
        self.feature_names = list(X_train.columns)
        
        start_time = time.time()
        
        fit_params = {}
        
        if self.model_type == 'xgboost':
            fit_params = {
                'eval_set': [(X_train, y_train)] if X_val is None else [(X_val, y_val)],
                'early_stopping_rounds': early_stopping_rounds,
                'verbose': False
            }
            
        elif self.model_type == 'lightgbm':
            fit_params = {
                'eval_set': [(X_val, y_val)] if X_val is not None else None,
                'early_stopping_rounds': early_stopping_rounds if X_val is not None else None,
                'verbose': False
            }
            
        elif self.model_type == 'catboost':
            fit_params = {
                'eval_set': (X_val, y_val) if X_val is not None else None,
                'early_stopping_rounds': early_stopping_rounds if X_val is not None else None,
                'verbose': False,
                'cat_features': categorical_features if categorical_features else []
            }
        
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train, **fit_params)
        
        self.training_time = time.time() - start_time
        
        # Get metrics
        metrics = self.evaluate(X_train, y_train)
        if X_val is not None:
            val_metrics = self.evaluate(X_val, y_val, prefix='val_')
            metrics.update(val_metrics)
        
        metrics['training_time_sec'] = self.training_time
        
        logger.info(f"Training completed in {self.training_time:.2f}s")
        logger.info(f"Validation AUC-ROC: {metrics.get('val_roc_auc', 'N/A')}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
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
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_time': self.training_time
        }, path)
        logger.info(f"Saved model to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.model_type = data['model_type']
        self.feature_names = data['feature_names']
        self.training_time = data.get('training_time', 0)
        logger.info(f"Loaded model from {path}")
        return self


class EnsembleFraudDetector:
    """
    Ensemble of multiple fraud detection models.
    Supports soft voting (probability averaging).
    """
    
    def __init__(self, models: Optional[List[FraudModelTrainer]] = None):
        self.models = models or []
        self.weights: Optional[np.ndarray] = None
        
    def add_model(self, model: FraudModelTrainer, weight: float = 1.0):
        """Add a trained model to the ensemble."""
        self.models.append(model)
        
    def fit_weights(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        method: str = 'roc_optimal'
    ):
        """Optimize ensemble weights based on validation performance."""
        
        if method == 'equal':
            self.weights = np.ones(len(self.models)) / len(self.models)
            
        elif method == 'roc_optimal':
            # Weight by ROC-AUC performance
            scores = []
            for model in self.models:
                y_prob = model.predict_proba(X)
                scores.append(roc_auc_score(y, y_prob))
            
            scores = np.array(scores)
            self.weights = scores / scores.sum()
            
        logger.info(f"Ensemble weights: {self.weights}")
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using weighted average of model probabilities."""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        if self.weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        probs = np.zeros(len(X))
        for model, weight in zip(self.models, self.weights):
            probs += model.predict_proba(X) * weight
        
        return probs
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        y_prob = self.predict_proba(X)
        y_pred = self.predict(X)
        
        return {
            'roc_auc': roc_auc_score(y, y_prob),
            'average_precision': average_precision_score(y, y_prob),
            'f1': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
        }


def train_multiple_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_types: List[str] = ['xgboost', 'lightgbm', 'catboost'],
    categorical_features: Optional[List[str]] = None
) -> Dict[str, FraudModelTrainer]:
    """Train multiple models and return trained models dict."""
    
    trained_models = {}
    
    for model_type in model_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_type.upper()}")
        logger.info(f"{'='*50}")
        
        trainer = FraudModelTrainer(model_type=model_type)
        trainer.build_model(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            class_weight='balanced'
        )
        
        metrics = trainer.fit(
            X_train, y_train,
            X_val, y_val,
            early_stopping_rounds=50,
            categorical_features=categorical_features
        )
        
        trained_models[model_type] = trainer
        
        logger.info(f"\n{model_type} Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")
    
    return trained_models
