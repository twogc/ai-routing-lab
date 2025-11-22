"""E2E tests for full workflow."""

import json

import numpy as np
import pytest

from models.core import ModelRegistry
from models.prediction import JitterPredictor, LatencyPredictor


@pytest.mark.e2e
class TestFullWorkflow:
    """End-to-end tests for complete ML workflow."""

    def test_full_training_and_prediction_workflow(self, tmp_path):
        """Test complete workflow from training to prediction."""

        # 1. Generate synthetic training data
        n_samples = 100
        n_features = 4

        X_train = np.random.randn(n_samples, n_features)
        y_latency = np.random.randn(n_samples) * 10 + 25
        y_jitter = np.random.randn(n_samples) * 2 + 2

        # 2. Train latency model
        latency_model = LatencyPredictor(n_estimators=10, max_depth=5)
        latency_model.fit(X_train, y_latency)

        # 3. Train jitter model
        jitter_model = JitterPredictor(n_estimators=10, max_depth=5)
        jitter_model.fit(X_train, y_jitter)

        # 4. Save models to registry
        registry = ModelRegistry(models_dir=str(tmp_path))

        registry.register_model(
            model_id="latency_predictor",
            model=latency_model,
            model_type="prediction",
            accuracy=0.95,
            framework="scikit-learn",
        )

        registry.register_model(
            model_id="jitter_predictor",
            model=jitter_model,
            model_type="prediction",
            accuracy=0.93,
            framework="scikit-learn",
        )

        # 5. Load models from registry
        loaded_latency, _ = registry.get_model("latency_predictor")
        loaded_jitter, _ = registry.get_model("jitter_predictor")

        # 6. Make predictions
        X_test = np.random.randn(5, n_features)

        latency_pred = loaded_latency.predict(X_test)
        jitter_pred = loaded_jitter.predict(X_test)

        # 7. Verify predictions
        assert latency_pred.predicted_latency_ms > 0
        assert jitter_pred.predicted_jitter_ms > 0
        assert 0 <= latency_pred.confidence_score <= 1
        assert 0 <= jitter_pred.confidence_score <= 1

    def test_data_collection_to_prediction(self, tmp_path):
        """Test workflow from data collection to prediction."""

        # This test would require actual quic-test running
        # For now, we'll test the structure

        # 1. Simulate collected data
        collected_data = [
            {"features": [25.5, 2.3, 0.95, 1.0], "latency": 26.0, "jitter": 2.5},
            {"features": [30.1, 3.1, 0.85, 1.2], "latency": 31.0, "jitter": 3.2},
        ] * 50  # Repeat to have enough data

        # 2. Save to file
        data_file = tmp_path / "training_data.json"
        with open(data_file, "w") as f:
            json.dump(collected_data, f)

        # 3. Load and prepare data
        with open(data_file, "r") as f:
            data = json.load(f)

        X = np.array([item["features"] for item in data])
        y_latency = np.array([item["latency"] for item in data])

        # 4. Train model
        model = LatencyPredictor(n_estimators=10)
        model.fit(X, y_latency)

        # 5. Make prediction
        test_features = np.array([[25.0, 2.0, 0.9, 1.0]])
        prediction = model.predict(test_features)

        assert prediction.predicted_latency_ms > 0
