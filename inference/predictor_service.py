"""
FastAPI service for model inference.

Provides REST API endpoints for route prediction and optimization.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from models.core.model_registry import ModelRegistry
from models.prediction.route_prediction_ensemble import RoutePredictionEnsemble

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_requests = Counter(
    "prediction_requests_total", "Total prediction requests", ["endpoint", "status"]
)

prediction_latency = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds", ["endpoint"]
)

# FastAPI app
app = FastAPI(
    title="AI Routing Lab Inference API",
    description="ML-based route prediction and optimization",
    version="0.2.1",
)

# Global model registry
registry: Optional[ModelRegistry] = None
ensemble_model: Optional[RoutePredictionEnsemble] = None


class PredictionRequest(BaseModel):
    """Request model for route prediction."""

    features: List[float] = Field(..., description="Feature vector for prediction", min_items=1)
    route_id: str = Field(..., description="Route identifier")


class PredictionResponse(BaseModel):
    """Response model for route prediction."""

    route_id: str
    predicted_latency_ms: float
    predicted_jitter_ms: float
    confidence_score: float
    prediction_time_ms: float


class RoutesRequest(BaseModel):
    """Request model for multiple routes comparison."""

    routes: Dict[str, List[float]] = Field(
        ..., description="Dictionary mapping route_id to feature vector"
    )


class RoutesResponse(BaseModel):
    """Response model for routes comparison."""

    best_route: str
    predictions: Dict[str, PredictionResponse]
    ranking: List[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: bool
    timestamp: float


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global registry, ensemble_model

    try:
        logger.info("Loading models...")
        models_dir = Path("models/")
        models_dir.mkdir(parents=True, exist_ok=True)

        registry = ModelRegistry(models_dir=str(models_dir))

        # Try to load ensemble model if it exists
        try:
            ensemble_model, _ = registry.get_model("route_ensemble")
            logger.info("Ensemble model loaded successfully")
        except KeyError:
            logger.warning("Ensemble model not found, will need to be trained")
            ensemble_model = None

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {"service": "AI Routing Lab Inference API", "version": "0.2.1", "status": "running"}


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy", models_loaded=ensemble_model is not None, timestamp=time.time()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_route(request: PredictionRequest):
    """
    Predict latency and jitter for a single route.

    Args:
        request: Prediction request with features and route_id

    Returns:
        Prediction response with latency, jitter, and confidence
    """
    start_time = time.time()

    try:
        if ensemble_model is None:
            prediction_requests.labels(endpoint="predict", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please train the model first.",
            )

        # Convert features to numpy array
        features = np.array(request.features).reshape(1, -1)

        # Make prediction
        prediction = ensemble_model.predict(features)

        prediction_time = (time.time() - start_time) * 1000  # Convert to ms

        # Record metrics
        prediction_requests.labels(endpoint="predict", status="success").inc()
        prediction_latency.labels(endpoint="predict").observe(time.time() - start_time)

        return PredictionResponse(
            route_id=request.route_id,
            predicted_latency_ms=float(prediction.predicted_latency_ms),
            predicted_jitter_ms=float(prediction.predicted_jitter_ms),
            confidence_score=float(prediction.confidence_score),
            prediction_time_ms=prediction_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        prediction_requests.labels(endpoint="predict", status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/routes", response_model=RoutesResponse, tags=["Prediction"])
async def predict_multiple_routes(request: RoutesRequest):
    """
    Compare multiple routes and return the best one.

    Args:
        request: Routes request with multiple route features

    Returns:
        Best route and predictions for all routes
    """
    start_time = time.time()

    try:
        if ensemble_model is None:
            prediction_requests.labels(endpoint="predict_routes", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please train the model first.",
            )

        predictions = {}
        scores = {}

        # Predict for each route
        for route_id, features in request.routes.items():
            features_array = np.array(features).reshape(1, -1)
            prediction = ensemble_model.predict(features_array)

            predictions[route_id] = PredictionResponse(
                route_id=route_id,
                predicted_latency_ms=float(prediction.predicted_latency_ms),
                predicted_jitter_ms=float(prediction.predicted_jitter_ms),
                confidence_score=float(prediction.confidence_score),
                prediction_time_ms=0.0,  # Will be set later
            )

            # Calculate combined score (lower is better)
            # Weighted: 70% latency, 30% jitter
            scores[route_id] = (
                0.7 * prediction.predicted_latency_ms + 0.3 * prediction.predicted_jitter_ms
            )

        # Find best route (lowest score)
        best_route = min(scores, key=scores.get)

        # Rank routes by score
        ranking = sorted(scores.keys(), key=lambda x: scores[x])

        prediction_time = (time.time() - start_time) * 1000

        # Update prediction times
        for pred in predictions.values():
            pred.prediction_time_ms = prediction_time / len(predictions)

        # Record metrics
        prediction_requests.labels(endpoint="predict_routes", status="success").inc()
        prediction_latency.labels(endpoint="predict_routes").observe(time.time() - start_time)

        return RoutesResponse(best_route=best_route, predictions=predictions, ranking=ranking)

    except HTTPException:
        raise
    except Exception as e:
        prediction_requests.labels(endpoint="predict_routes", status="error").inc()
        logger.error(f"Routes prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Routes prediction failed: {str(e)}",
        )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models", tags=["Models"])
async def list_models():
    """List all loaded models."""
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model registry not initialized"
        )

    models = registry.list_models()
    return {
        "models": [
            {
                "model_id": metadata.model_id,
                "model_type": metadata.model_type,
                "version": metadata.version,
                "accuracy": metadata.accuracy,
                "framework": metadata.framework,
            }
            for metadata in models
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
