"""Unit tests for LatencyPredictor model."""

import numpy as np
import pytest

from models.prediction.latency_predictor import LatencyPrediction, LatencyPredictor


@pytest.mark.unit
class TestLatencyPredictor:
    """Test suite for LatencyPredictor."""

    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = LatencyPredictor()

        assert model.n_estimators == 100
        assert model.max_depth == 15
        assert model.random_state == 42
        assert model.model is not None
        assert model.scaler is not None

    def test_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = LatencyPredictor(n_estimators=50, max_depth=10, random_state=123)

        assert model.n_estimators == 50
        assert model.max_depth == 10
        assert model.random_state == 123

    def test_fit(self, sample_features, sample_latency_targets):
        """Test model training."""
        model = LatencyPredictor()
        result = model.fit(sample_features, sample_latency_targets)

        # Check that fit returns self
        assert result is model

        # Check that model is fitted
        assert hasattr(model.model, "estimators_")

        # Check that scaler is fitted
        assert hasattr(model.scaler, "mean_")

    def test_fit_with_feature_names(self, sample_features, sample_latency_targets):
        """Test model training with feature names."""
        feature_names = ["avg_latency", "variance", "stability", "load"]
        model = LatencyPredictor()
        model.fit(sample_features, sample_latency_targets, feature_names=feature_names)

        assert model.feature_names == feature_names

    def test_predict_without_fit_raises_error(self, sample_features):
        """Test that prediction without fitting raises error."""
        model = LatencyPredictor()

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.predict(sample_features)

    def test_predict_single_sample(self, sample_features, sample_latency_targets):
        """Test prediction on single sample."""
        model = LatencyPredictor()
        model.fit(sample_features, sample_latency_targets)

        # Predict on single sample
        single_sample = sample_features[0:1]
        prediction = model.predict(single_sample)

        assert isinstance(prediction, LatencyPrediction)
        assert prediction.predicted_latency_ms > 0
        assert len(prediction.confidence_interval) == 2
        assert 0 <= prediction.confidence_score <= 1

    def test_predict_multiple_samples(self, sample_features, sample_latency_targets):
        """Test prediction on multiple samples."""
        model = LatencyPredictor()
        model.fit(sample_features, sample_latency_targets)

        # For multiple samples, predict returns a single LatencyPrediction with mean
        predictions = model.predict(sample_features)

        assert isinstance(predictions, LatencyPrediction)
        assert isinstance(predictions.predicted_latency_ms, (int, float))
        assert predictions.predicted_latency_ms > 0

    def test_predict_with_confidence(self, sample_features, sample_latency_targets):
        """Test prediction with confidence intervals."""
        model = LatencyPredictor()
        model.fit(sample_features, sample_latency_targets)

        prediction = model.predict(sample_features[0:1], return_confidence=True)

        assert prediction.confidence_interval is not None
        assert prediction.confidence_interval[0] < prediction.predicted_latency_ms
        assert prediction.confidence_interval[1] > prediction.predicted_latency_ms

    def test_predict_without_confidence(self, sample_features, sample_latency_targets):
        """Test prediction without confidence intervals."""
        model = LatencyPredictor()
        model.fit(sample_features, sample_latency_targets)

        prediction = model.predict(sample_features[0:1], return_confidence=False)

        assert prediction.predicted_latency_ms > 0

    def test_evaluate(self, sample_features, sample_latency_targets):
        """Test model evaluation."""
        model = LatencyPredictor()
        model.fit(sample_features, sample_latency_targets)

        metrics = model.evaluate(sample_features, sample_latency_targets)

        assert "r2_score" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics

        # R² should be between -inf and 1 (typically > 0 for good models)
        assert metrics["r2_score"] <= 1.0

        # MAE and RMSE should be positive
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

    def test_get_metrics(self, sample_features, sample_latency_targets):
        """Test getting model metrics."""
        model = LatencyPredictor()
        model.fit(sample_features, sample_latency_targets)
        model.evaluate(sample_features, sample_latency_targets)

        metrics = model.get_metrics()

        assert metrics is not None
        assert "r2_score" in metrics

    def test_get_feature_importance(self, sample_features, sample_latency_targets):
        """Test getting feature importance."""
        feature_names = ["avg_latency", "variance", "stability", "load"]
        model = LatencyPredictor()
        model.fit(sample_features, sample_latency_targets, feature_names=feature_names)

        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == len(feature_names)
        assert all(name in importance for name in feature_names)
        assert all(0 <= val <= 1 for val in importance.values())

    def test_ensemble_prediction(self, sample_features, sample_latency_targets):
        """Test ensemble prediction with gradient boosting."""
        model = LatencyPredictor(use_gradient_boosting=True)
        model.fit(sample_features, sample_latency_targets, use_ensemble=True)

        prediction = model.predict(sample_features[0:1], use_ensemble=True)

        assert prediction.predicted_latency_ms > 0

    def test_prediction_time(self, sample_features, sample_latency_targets):
        """Test that prediction time is recorded."""
        model = LatencyPredictor()
        model.fit(sample_features, sample_latency_targets)

        prediction = model.predict(sample_features[0:1])

        assert prediction.prediction_time_ms >= 0

    def test_invalid_input_shape(self, sample_latency_targets):
        """Test that invalid input shape raises error."""
        model = LatencyPredictor()

        # Wrong shape
        invalid_features = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            model.fit(invalid_features, sample_latency_targets)

    def test_mismatched_samples(self, sample_features):
        """Test that mismatched X and y samples raises error."""
        model = LatencyPredictor()

        # Different number of samples
        wrong_targets = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            model.fit(sample_features, wrong_targets)

    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test model performance on larger dataset."""
        # Generate larger dataset
        n_samples = 1000
        n_features = 4

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples) * 10 + 25  # Latency around 25ms

        model = LatencyPredictor()
        model.fit(X, y)

        metrics = model.evaluate(X, y)

        # Model should achieve reasonable performance
        assert metrics["r2_score"] > 0.5
