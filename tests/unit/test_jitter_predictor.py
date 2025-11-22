"""Unit tests for JitterPredictor model."""

import pytest

from models.prediction.jitter_predictor import JitterPrediction, JitterPredictor


@pytest.mark.unit
class TestJitterPredictor:
    """Test suite for JitterPredictor."""

    def test_initialization(self):
        """Test model initialization."""
        model = JitterPredictor()

        assert model.n_estimators == 100
        assert model.max_depth == 15
        assert model.model is not None

    def test_fit(self, sample_features, sample_jitter_targets):
        """Test model training."""
        model = JitterPredictor()
        result = model.fit(sample_features, sample_jitter_targets)

        assert result is model
        assert hasattr(model.model, "estimators_")

    def test_predict(self, sample_features, sample_jitter_targets):
        """Test jitter prediction."""
        model = JitterPredictor()
        model.fit(sample_features, sample_jitter_targets)

        prediction = model.predict(sample_features[0:1])

        assert isinstance(prediction, JitterPrediction)
        assert prediction.predicted_jitter_ms > 0
        assert 0 <= prediction.confidence_score <= 1

    def test_evaluate(self, sample_features, sample_jitter_targets):
        """Test model evaluation."""
        model = JitterPredictor()
        model.fit(sample_features, sample_jitter_targets)

        metrics = model.evaluate(sample_features, sample_jitter_targets)

        assert "r2_score" in metrics
        assert "mae" in metrics
        assert metrics["mae"] >= 0

    def test_predict_without_fit(self, sample_features):
        """Test prediction without fitting raises error."""
        model = JitterPredictor()

        with pytest.raises(RuntimeError):
            model.predict(sample_features)
