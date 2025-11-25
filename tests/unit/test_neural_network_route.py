"""Unit tests for NeuralNetworkRouteOptimizer."""

import numpy as np
import pytest

from models.routing.neural_network_route import NeuralNetworkRouteOptimizer, NNRouteSelection


class TestNeuralNetworkRouteOptimizer:
    """Test suite for NeuralNetworkRouteOptimizer."""

    @pytest.fixture
    def optimizer(self):
        return NeuralNetworkRouteOptimizer(n_routes=3, hidden_dim=10)

    @pytest.fixture
    def sample_data(self):
        X = np.random.rand(10, 4)
        y = np.random.randint(0, 3, 10)
        return X, y

    def test_initialization(self, optimizer):
        """Test initialization."""
        assert optimizer.n_routes == 3
        assert optimizer.hidden_dim == 10
        assert not optimizer.fitted
        assert optimizer.weights1 is None

    def test_fit(self, optimizer, sample_data):
        """Test training (mock implementation)."""
        X, y = sample_data
        optimizer.fit(X, y, epochs=2)

        assert optimizer.fitted
        assert optimizer.weights1 is not None
        assert optimizer.weights2 is not None
        assert optimizer.weights1.shape == (4, 10)
        assert optimizer.weights2.shape == (10, 3)

    def test_predict_not_fitted(self, optimizer, sample_data):
        """Test prediction before training."""
        X, _ = sample_data
        with pytest.raises(RuntimeError):
            optimizer.predict(X)

    def test_predict(self, optimizer, sample_data):
        """Test prediction."""
        X, y = sample_data
        optimizer.fit(X, y)

        preds, confs = optimizer.predict(X)
        assert len(preds) == len(X)
        assert len(confs) == len(X)
        assert np.all((preds >= 0) & (preds < 3))
        assert np.all((confs >= 0) & (confs <= 1))

    def test_predict_sample(self, optimizer, sample_data):
        """Test single sample prediction."""
        X, y = sample_data
        optimizer.fit(X, y)

        selection = optimizer.predict_sample(X[0])
        assert isinstance(selection, NNRouteSelection)
        assert 0 <= selection.selected_route < 3
        assert len(selection.probabilities) == 3
        assert len(selection.top_routes) == 3

    def test_score(self, optimizer, sample_data):
        """Test scoring."""
        X, y = sample_data
        optimizer.fit(X, y)

        score = optimizer.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_get_metrics(self, optimizer):
        """Test metrics retrieval."""
        metrics = optimizer.get_metrics()
        assert "n_routes" in metrics
        assert "hidden_dim" in metrics

    def test_softmax(self, optimizer):
        """Test softmax function."""
        x = np.array([[1.0, 2.0, 3.0]])
        probs = optimizer._softmax(x)
        assert np.allclose(np.sum(probs), 1.0)
        assert probs[0, 2] > probs[0, 1] > probs[0, 0]
