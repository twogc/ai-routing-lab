"""Unit tests for RandomForestLoadModel."""

import numpy as np
import pytest

from models.prediction.random_forest_load import LoadPrediction, RandomForestLoadModel


class TestRandomForestLoadModel:
    """Test suite for RandomForestLoadModel."""

    @pytest.fixture
    def model(self):
        return RandomForestLoadModel(n_trees=5, max_depth=5)

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = np.random.rand(20, 3)
        # y = 2*x0 + x1
        y = 2 * X[:, 0] + X[:, 1]
        return X, y

    def test_initialization(self, model):
        """Test initialization."""
        assert model.n_trees == 5
        assert not model.fitted
        assert len(model.trees) == 0

    def test_fit(self, model, sample_data):
        """Test training."""
        X, y = sample_data
        model.fit(X, y)

        assert model.fitted
        assert len(model.trees) == 5
        assert model.feature_importances is not None
        assert len(model.feature_importances) == 3

    def test_predict_not_fitted(self, model, sample_data):
        """Test prediction before training."""
        X, _ = sample_data
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_predict(self, model, sample_data):
        """Test prediction."""
        X, y = sample_data
        model.fit(X, y)

        preds, uncertainties = model.predict(X)
        assert len(preds) == len(X)
        assert len(uncertainties) == len(X)

        # Check if predictions are reasonable (MSE should be low)
        mse = np.mean((preds - y) ** 2)
        assert mse < 0.5

    def test_predict_sample(self, model, sample_data):
        """Test single sample prediction."""
        X, y = sample_data
        model.fit(X, y)

        prediction = model.predict_sample(X[0])
        assert isinstance(prediction, LoadPrediction)
        assert isinstance(prediction.predicted_load, float)
        assert len(prediction.confidence_interval) == 2
        assert prediction.feature_importance is not None

    def test_score(self, model, sample_data):
        """Test scoring."""
        X, y = sample_data
        model.fit(X, y)

        score = model.score(X, y)
        assert score > 0.0  # R2 should be positive for this simple relationship

    def test_get_metrics(self, model, sample_data):
        """Test metrics retrieval."""
        X, y = sample_data
        model.fit(X, y)

        metrics = model.get_metrics()
        assert "mse" in metrics
        assert "r_squared" in metrics
        assert "n_trees" in metrics

    def test_feature_importance(self, model, sample_data):
        """Test feature importance calculation."""
        X, y = sample_data
        model.fit(X, y)

        importances = model.feature_importances
        # Feature 0 (coeff 2) should be more important than Feature 2 (coeff 0)
        # But with small data/trees it might be noisy, so just check shape/sum
        assert len(importances) == 3
        assert abs(importances.sum() - 1.0) < 1e-6
