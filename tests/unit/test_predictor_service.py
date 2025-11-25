"""Unit tests for Inference Service."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

from inference.predictor_service import app
from models.prediction.route_prediction_ensemble import RoutePredictionEnsemble, RoutePrediction

client = TestClient(app)


@pytest.fixture
def mock_ensemble_model():
    """Mock RoutePredictionEnsemble."""
    mock_model = MagicMock(spec=RoutePredictionEnsemble)

    # Mock predict method
    mock_prediction = RoutePrediction(
        route_id=0,
        predicted_latency_ms=25.5,
        predicted_jitter_ms=2.3,
        combined_score=0.5,
        latency_confidence=0.95,
        jitter_confidence=0.95,
        overall_confidence=0.95,
    )
    mock_model.predict.return_value = mock_prediction
    return mock_model


@pytest.fixture
def mock_registry():
    """Mock ModelRegistry."""
    mock_reg = MagicMock()
    mock_reg.list_models.return_value = []
    return mock_reg


class TestPredictorService:
    """Test suite for Predictor Service."""

    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "running"

    @patch("inference.predictor_service.ensemble_model")
    def test_health_check(self, mock_model):
        """Test health check endpoint."""
        # Case 1: Model loaded
        mock_model.__bool__.return_value = True
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["models_loaded"] is True

        # Case 2: Model not loaded
        with patch("inference.predictor_service.ensemble_model", None):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["models_loaded"] is False

    @patch("inference.predictor_service.ensemble_model")
    def test_predict_route(self, mock_model, mock_ensemble_model):
        """Test single route prediction."""
        # Setup mock
        mock_model.predict = mock_ensemble_model.predict

        payload = {"features": [25.5, 2.3, 0.95, 1.0], "route_id": "route_0"}

        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["route_id"] == "route_0"
        assert data["predicted_latency_ms"] == 25.5
        assert data["predicted_jitter_ms"] == 2.3
        assert data["confidence_score"] == 0.95

    def test_predict_route_no_model(self):
        """Test prediction when model is not loaded."""
        with patch("inference.predictor_service.ensemble_model", None):
            payload = {"features": [25.5, 2.3, 0.95, 1.0], "route_id": "route_0"}
            response = client.post("/predict", json=payload)
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]

    @patch("inference.predictor_service.ensemble_model")
    def test_predict_multiple_routes(self, mock_model, mock_ensemble_model):
        """Test multiple routes prediction."""
        # Setup mock
        mock_model.predict = mock_ensemble_model.predict

        payload = {"routes": {"route_0": [25.5, 2.3, 0.95, 1.0], "route_1": [30.1, 3.1, 0.85, 1.2]}}

        response = client.post("/predict/routes", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "best_route" in data
        assert "predictions" in data
        assert "ranking" in data
        assert len(data["predictions"]) == 2

    def test_metrics(self):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "prediction_requests_total" in response.text

    @patch("inference.predictor_service.registry")
    def test_list_models(self, mock_reg):
        """Test list models endpoint."""
        # Setup mock
        mock_metadata = MagicMock()
        mock_metadata.model_id = "test_model"
        mock_metadata.model_type = "prediction"
        mock_metadata.version = "1.0.0"
        mock_metadata.accuracy = 0.95
        mock_metadata.framework = "sklearn"

        mock_reg.list_models.return_value = [mock_metadata]

        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert len(data["models"]) == 1
        assert data["models"][0]["model_id"] == "test_model"

    def test_list_models_no_registry(self):
        """Test list models when registry is not initialized."""
        with patch("inference.predictor_service.registry", None):
            response = client.get("/models")
            assert response.status_code == 503
