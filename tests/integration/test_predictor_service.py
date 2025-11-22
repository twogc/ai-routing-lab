"""Integration tests for predictor service."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from inference.predictor_service import app
from models.core.model_registry import ModelRegistry
from models.prediction.jitter_predictor import JitterPredictor
from models.prediction.latency_predictor import LatencyPredictor
from models.prediction.route_prediction_ensemble import RoutePredictionEnsemble


@pytest.fixture
def temp_models_dir():
    """Temporary directory for models."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def trained_ensemble(temp_models_dir):
    """Create and train an ensemble model."""
    # Generate training data
    X_train = np.random.randn(50, 4)
    y_latency = np.random.randn(50) * 10 + 25
    y_jitter = np.random.randn(50) * 2 + 2

    # Train models
    latency_model = LatencyPredictor(n_estimators=10, max_depth=5)
    latency_model.fit(X_train, y_latency)

    jitter_model = JitterPredictor(n_estimators=10, max_depth=5)
    jitter_model.fit(X_train, y_jitter)

    # Create ensemble
    ensemble = RoutePredictionEnsemble(latency_model=latency_model, jitter_model=jitter_model)

    # Register in registry
    registry = ModelRegistry(models_dir=str(temp_models_dir))
    registry.register_model(
        model_id="route_ensemble",
        model=ensemble,
        model_type="routing",
        accuracy=0.95,
        framework="scikit-learn",
    )

    return ensemble, temp_models_dir


@pytest.mark.integration
@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestPredictorService:
    """Integration tests for predictor service."""

    def test_root_endpoint(self):
        """Test root endpoint."""
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "AI Routing Lab Inference API"
        assert data["status"] == "running"

    def test_health_check(self):
        """Test health check endpoint."""
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "timestamp" in data

    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        client = TestClient(app)
        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        assert "prediction_requests_total" in response.text

    def test_predict_endpoint_no_model(self):
        """Test predict endpoint when model is not loaded."""
        client = TestClient(app)

        request_data = {"features": [25.5, 2.3, 0.95, 1.0], "route_id": "route_0"}

        response = client.post("/predict", json=request_data)

        # Should return 503 Service Unavailable
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_predict_endpoint_invalid_request(self):
        """Test predict endpoint with invalid request."""
        client = TestClient(app)

        # Missing required fields
        response = client.post("/predict", json={})

        assert response.status_code == 422  # Validation error

    def test_list_models_endpoint(self):
        """Test list models endpoint."""
        client = TestClient(app)
        response = client.get("/models")

        # Should work even if no models are loaded
        assert response.status_code in [200, 503]

    def test_predict_routes_endpoint_no_model(self):
        """Test predict/routes endpoint when model is not loaded."""
        client = TestClient(app)

        request_data = {
            "routes": {"route_0": [25.5, 2.3, 0.95, 1.0], "route_1": [30.1, 3.1, 0.85, 1.2]}
        }

        response = client.post("/predict/routes", json=request_data)

        # Should return 503 Service Unavailable
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_predict_routes_endpoint_invalid_request(self):
        """Test predict/routes endpoint with invalid request."""
        client = TestClient(app)

        # Empty routes
        response = client.post("/predict/routes", json={"routes": {}})

        assert response.status_code == 422  # Validation error


@pytest.mark.integration
@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestPredictorServiceWithModel:
    """Integration tests with loaded model."""

    # Note: These tests would require mocking the startup event
    # or using a test fixture that loads the model

    def test_service_structure(self):
        """Test that service has required endpoints."""
        client = TestClient(app)

        # Test all endpoints exist
        endpoints = ["/", "/health", "/metrics", "/models"]

        for endpoint in endpoints:
            response = client.get(endpoint)
            # Should not return 404
            assert response.status_code != 404, f"Endpoint {endpoint} not found"
