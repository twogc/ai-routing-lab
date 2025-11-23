"""Unit tests for IsolationForestModel."""

import pytest
import numpy as np
from models.anomaly.isolation_forest import IsolationForestModel, IsolationForestEnsemble, AnomalyPrediction

class TestIsolationForestModel:
    """Test suite for IsolationForestModel."""

    @pytest.fixture
    def model(self):
        return IsolationForestModel(n_trees=10, sample_size=20)

    @pytest.fixture
    def sample_data(self):
        # Normal data: clustered around 0
        X_normal = np.random.normal(0, 0.5, (50, 2))
        # Anomalies: far from 0
        X_anomaly = np.random.normal(5, 0.5, (5, 2))
        X = np.vstack([X_normal, X_anomaly])
        y = np.hstack([np.zeros(50), np.ones(5)])
        return X, y

    def test_initialization(self, model):
        """Test initialization."""
        assert model.n_trees == 10
        assert not model.fitted
        assert len(model.trees) == 0

    def test_fit(self, model, sample_data):
        """Test training."""
        X, _ = sample_data
        model.fit(X)
        
        assert model.fitted
        assert len(model.trees) == 10
        assert model.thresholds is not None

    def test_predict_not_fitted(self, model, sample_data):
        """Test prediction before training."""
        X, _ = sample_data
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_predict(self, model, sample_data):
        """Test prediction."""
        X, y = sample_data
        model.fit(X)
        
        preds, scores = model.predict(X)
        assert len(preds) == len(X)
        assert len(scores) == len(X)
        # Should detect some anomalies
        assert np.sum(preds) > 0

    def test_predict_sample(self, model, sample_data):
        """Test single sample prediction."""
        X, _ = sample_data
        model.fit(X)
        
        prediction = model.predict_sample(X[0])
        assert isinstance(prediction, AnomalyPrediction)
        assert 0.0 <= prediction.confidence <= 1.0
        assert 0.0 <= prediction.anomaly_score <= 1.0

    def test_score(self, model, sample_data):
        """Test scoring."""
        X, y = sample_data
        model.fit(X)
        
        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_get_metrics(self, model, sample_data):
        """Test metrics retrieval."""
        X, _ = sample_data
        model.fit(X)
        model.predict(X)
        
        metrics = model.get_metrics()
        assert "anomaly_rate_percent" in metrics
        assert "n_trees" in metrics

class TestIsolationForestEnsemble:
    """Test suite for IsolationForestEnsemble."""

    def test_ensemble(self):
        """Test ensemble functionality."""
        X = np.random.normal(0, 1, (20, 2))
        ensemble = IsolationForestEnsemble(n_models=2, n_trees=5, sample_size=10)
        
        ensemble.fit(X)
        preds, scores = ensemble.predict(X)
        
        assert len(preds) == 20
        assert len(scores) == 20
        assert len(ensemble.models) == 2
