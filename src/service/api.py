"""
FastAPI-based real-time explanation service.
Provides endpoints for fraud prediction and explanation.
"""
import logging
import time
import asyncio
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available")

from .caching_layer import ExplanationCache, RedisCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class TransactionFeatures(BaseModel):
    """Transaction feature schema."""
    TransactionAmt: float = Field(..., description="Transaction amount")
    card1: Optional[float] = Field(None, description="Card feature 1")
    card2: Optional[float] = Field(None, description="Card feature 2")
    card3: Optional[float] = Field(None, description="Card feature 3")
    card5: Optional[float] = Field(None, description="Card feature 5")
    addr1: Optional[float] = Field(None, description="Address 1")
    addr2: Optional[float] = Field(None, description="Address 2")
    dist1: Optional[float] = Field(None, description="Distance 1")
    dist2: Optional[float] = Field(None, description="Distance 2")
    C1: Optional[float] = Field(None, description="Count feature 1")
    C2: Optional[float] = Field(None, description="Count feature 2")
    C3: Optional[float] = Field(None, description="Count feature 3")
    C4: Optional[float] = Field(None, description="Count feature 4")
    C5: Optional[float] = Field(None, description="Count feature 5")
    C6: Optional[float] = Field(None, description="Count feature 6")
    C7: Optional[float] = Field(None, description="Count feature 7")
    C8: Optional[float] = Field(None, description="Count feature 8")
    C9: Optional[float] = Field(None, description="Count feature 9")
    C10: Optional[float] = Field(None, description="Count feature 10")
    C11: Optional[float] = Field(None, description="Count feature 11")
    C12: Optional[float] = Field(None, description="Count feature 12")
    C13: Optional[float] = Field(None, description="Count feature 13")
    C14: Optional[float] = Field(None, description="Count feature 14")
    D1: Optional[float] = Field(None, description="Time delta 1")
    D2: Optional[float] = Field(None, description="Time delta 2")
    D3: Optional[float] = Field(None, description="Time delta 3")
    D4: Optional[float] = Field(None, description="Time delta 4")
    D5: Optional[float] = Field(None, description="Time delta 5")
    D6: Optional[float] = Field(None, description="Time delta 6")
    D7: Optional[float] = Field(None, description="Time delta 7")
    D8: Optional[float] = Field(None, description="Time delta 8")
    D9: Optional[float] = Field(None, description="Time delta 9")
    D10: Optional[float] = Field(None, description="Time delta 10")
    D11: Optional[float] = Field(None, description="Time delta 11")
    D12: Optional[float] = Field(None, description="Time delta 12")
    D13: Optional[float] = Field(None, description="Time delta 13")
    D14: Optional[float] = Field(None, description="Time delta 14")
    D15: Optional[float] = Field(None, description="Time delta 15")
    V1: Optional[float] = Field(None, description="Vesta feature 1")
    V2: Optional[float] = Field(None, description="Vesta feature 2")
    V3: Optional[float] = Field(None, description="Vesta feature 3")
    V4: Optional[float] = Field(None, description="Vesta feature 4")
    V5: Optional[float] = Field(None, description="Vesta feature 5")
    V6: Optional[float] = Field(None, description="Vesta feature 6")
    V7: Optional[float] = Field(None, description="Vesta feature 7")
    V8: Optional[float] = Field(None, description="Vesta feature 8")
    V9: Optional[float] = Field(None, description="Vesta feature 9")
    V10: Optional[float] = Field(None, description="Vesta feature 10")
    V11: Optional[float] = Field(None, description="Vesta feature 11")
    V12: Optional[float] = Field(None, description="Vesta feature 12")
    V13: Optional[float] = Field(None, description="Vesta feature 13")
    V14: Optional[float] = Field(None, description="Vesta feature 14")
    V15: Optional[float] = Field(None, description="Vesta feature 15")
    V16: Optional[float] = Field(None, description="Vesta feature 16")
    V17: Optional[float] = Field(None, description="Vesta feature 17")
    V18: Optional[float] = Field(None, description="Vesta feature 18")
    V19: Optional[float] = Field(None, description="Vesta feature 19")
    V20: Optional[float] = Field(None, description="Vesta feature 20")


class BatchTransactionRequest(BaseModel):
    """Batch transaction request."""
    transactions: List[TransactionFeatures]
    explanation_method: str = Field(default='fastshap', description='Explanation method')
    return_top_k: int = Field(default=10, description='Number of top features to return')


class ExplanationResponse(BaseModel):
    """Explanation response schema."""
    transaction_id: Optional[str] = None
    fraud_probability: float
    prediction: int
    explanation_method: str
    shap_values: Dict[str, float]
    top_features: List[Dict[str, Any]]
    expected_value: float
    latency_ms: float
    from_cache: bool = False


class BatchExplanationResponse(BaseModel):
    """Batch explanation response."""
    results: List[ExplanationResponse]
    total_latency_ms: float
    average_latency_ms: float
    cache_hit_rate: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    explainer_loaded: bool
    cache_stats: Dict[str, Any]


class FraudExplanationService:
    """
    Core service for fraud detection and explanation.
    """
    
    def __init__(
        self,
        model=None,
        explainer=None,
        preprocessor=None,
        fallback_explainer=None,
        latency_threshold_ms: float = 100.0
    ):
        self.model = model
        self.explainer = explainer
        self.preprocessor = preprocessor
        self.fallback_explainer = fallback_explainer
        self.latency_threshold_ms = latency_threshold_ms
        
        self.cache = ExplanationCache()
        self.request_count = 0
        self.error_count = 0
        self.latency_history: List[float] = []
        
        self._feature_names: List[str] = []
        self._precomputed_importance: Optional[Dict] = None
        
    def load_model(self, model_path: str):
        """Load trained model."""
        import joblib
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
    def load_explainer(self, explainer_path: str):
        """Load FastSHAP explainer."""
        from ..explainers.fastshap_implementation import FastSHAPExplainer
        self.explainer = FastSHAPExplainer(input_dim=1)  # Placeholder
        self.explainer.load(explainer_path)
        logger.info(f"Loaded explainer from {explainer_path}")
        
    def set_precomputed_importance(self, importance: Dict):
        """Set pre-computed feature importance for fallback."""
        self._precomputed_importance = importance
        
    def preprocess(self, features: Dict) -> np.ndarray:
        """Preprocess transaction features."""
        df = pd.DataFrame([features])
        
        if self.preprocessor:
            df = self.preprocessor.transform(df)
        
        # Ensure consistent feature order
        if self._feature_names:
            df = df.reindex(columns=self._feature_names, fill_value=0)
        
        return df.values.astype(np.float32)
    
    def predict(self, X: np.ndarray) -> Dict:
        """Generate fraud prediction."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        prob = self.model.predict_proba(X)[0, 1]
        pred = int(prob >= 0.5)
        
        return {
            'fraud_probability': float(prob),
            'prediction': pred
        }
    
    def explain(
        self,
        X: np.ndarray,
        method: str = 'fastshap',
        use_cache: bool = True
    ) -> Dict:
        """
        Generate explanation with caching and fallback.
        
        Args:
            X: Preprocessed features
            method: Explanation method ('fastshap', 'treesha', 'fallback')
            use_cache: Whether to use caching
            
        Returns:
            Explanation result
        """
        def compute_explanation():
            start = time.time()
            
            try:
                if method == 'fastshap' and self.explainer is not None:
                    result = self.explainer.explain_single(X)
                elif method == 'treesha':
                    from ..explainers.baseline_shap import TreeSHAPExplainer
                    tree_exp = TreeSHAPExplainer(self.model)
                    tree_exp.fit()
                    result = tree_exp.explain_single(X)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                latency = (time.time() - start) * 1000
                result['latency_ms'] = latency
                
                # Check latency threshold
                if latency > self.latency_threshold_ms:
                    logger.warning(f"Latency {latency:.1f}ms exceeded threshold, using fallback")
                    return self._fallback_explanation(X)
                
                return result
                
            except Exception as e:
                logger.error(f"Explanation error: {e}")
                return self._fallback_explanation(X)
        
        return self.cache.get_or_compute(
            X, compute_explanation, method, use_cache
        )
    
    def _fallback_explanation(self, X: np.ndarray) -> Dict:
        """Generate fallback explanation using pre-computed importance."""
        if self._precomputed_importance:
            # Use pre-computed global importance
            shap_values = np.array([
                self._precomputed_importance.get(f, 0.0) 
                for f in self._feature_names
            ])
            # Scale by prediction probability
            prob = self.model.predict_proba(X)[0, 1]
            shap_values = shap_values * (prob - 0.5)
        else:
            shap_values = np.zeros(len(self._feature_names))
        
        return {
            'shap_values': shap_values,
            'expected_value': 0.0,
            'latency_ms': 1.0,  # Very fast
            'is_fallback': True
        }
    
    def process_transaction(
        self,
        features: Dict,
        explanation_method: str = 'fastshap',
        return_top_k: int = 10,
        transaction_id: Optional[str] = None
    ) -> ExplanationResponse:
        """
        Process a single transaction with prediction and explanation.
        
        Args:
            features: Transaction features
            explanation_method: Explanation method
            return_top_k: Number of top features to return
            transaction_id: Optional transaction ID
            
        Returns:
            ExplanationResponse
        """
        self.request_count += 1
        
        try:
            # Preprocess
            X = self.preprocess(features)
            
            # Predict
            pred_result = self.predict(X)
            
            # Explain
            exp_result = self.explain(X, explanation_method)
            
            # Format SHAP values
            shap_values = exp_result['shap_values']
            if isinstance(shap_values, np.ndarray):
                shap_values = shap_values.flatten()
            
            shap_dict = {
                name: float(val) 
                for name, val in zip(self._feature_names, shap_values)
            }
            
            # Get top features
            top_indices = np.argsort(np.abs(shap_values))[-return_top_k:][::-1]
            top_features = [
                {
                    'feature': self._feature_names[i],
                    'shap_value': float(shap_values[i]),
                    'importance_rank': rank + 1
                }
                for rank, i in enumerate(top_indices)
            ]
            
            self.latency_history.append(exp_result['latency_ms'])
            
            return ExplanationResponse(
                transaction_id=transaction_id,
                fraud_probability=pred_result['fraud_probability'],
                prediction=pred_result['prediction'],
                explanation_method=explanation_method,
                shap_values=shap_dict,
                top_features=top_features,
                expected_value=float(exp_result.get('expected_value', 0.0)),
                latency_ms=exp_result['latency_ms'],
                from_cache=exp_result.get('from_cache', False)
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error processing transaction: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_stats(self) -> Dict:
        """Get service statistics."""
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'average_latency_ms': np.mean(self.latency_history) if self.latency_history else 0,
            'p95_latency_ms': np.percentile(self.latency_history, 95) if self.latency_history else 0,
            'cache_stats': self.cache.get_stats()
        }


# Global service instance
_service: Optional[FraudExplanationService] = None


def get_service() -> FraudExplanationService:
    """Get or create global service instance."""
    global _service
    if _service is None:
        _service = FraudExplanationService()
    return _service


if FASTAPI_AVAILABLE:
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup/shutdown."""
        # Startup
        logger.info("Starting up explanation service...")
        yield
        # Shutdown
        logger.info("Shutting down explanation service...")
    
    
    def create_app(
        model_path: Optional[str] = None,
        explainer_path: Optional[str] = None
    ) -> FastAPI:
        """Create FastAPI application."""
        
        app = FastAPI(
            title="Real-Time Fraud XAI Service",
            description="FastSHAP-based real-time explanation service for fraud detection",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize service
        service = get_service()
        if model_path:
            service.load_model(model_path)
        if explainer_path:
            service.load_explainer(explainer_path)
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                model_loaded=service.model is not None,
                explainer_loaded=service.explainer is not None,
                cache_stats=service.cache.get_stats()
            )
        
        @app.post("/predict", response_model=ExplanationResponse)
        async def predict_and_explain(
            features: TransactionFeatures,
            explanation_method: str = Query(default='fastshap', enum=['fastshap', 'treesha', 'fallback']),
            return_top_k: int = Query(default=10, ge=1, le=50),
            transaction_id: Optional[str] = None
        ):
            """
            Predict fraud probability and generate explanation.
            
            - **explanation_method**: fastshap (fast), treesha (exact), fallback (pre-computed)
            - **return_top_k**: Number of top features to return (1-50)
            """
            return service.process_transaction(
                features=dict(features),
                explanation_method=explanation_method,
                return_top_k=return_top_k,
                transaction_id=transaction_id
            )
        
        @app.post("/explain/batch", response_model=BatchExplanationResponse)
        async def explain_batch(request: BatchTransactionRequest):
            """Process multiple transactions in batch."""
            start_time = time.time()
            
            results = []
            for i, tx in enumerate(request.transactions):
                result = service.process_transaction(
                    features=dict(tx),
                    explanation_method=request.explanation_method,
                    return_top_k=request.return_top_k,
                    transaction_id=f"batch_{i}"
                )
                results.append(result)
            
            total_latency = (time.time() - start_time) * 1000
            
            return BatchExplanationResponse(
                results=results,
                total_latency_ms=total_latency,
                average_latency_ms=total_latency / len(request.transactions),
                cache_hit_rate=service.cache.get_stats()['hit_rate']
            )
        
        @app.get("/stats")
        async def get_service_stats():
            """Get service statistics."""
            return service.get_stats()
        
        @app.post("/cache/invalidate")
        async def invalidate_cache(pattern: str = "*"):
            """Invalidate cache entries."""
            count = service.cache.cache.invalidate(pattern)
            return {"invalidated_entries": count}
        
        return app

else:
    def create_app(*args, **kwargs):
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
