"""Unit tests for MultiArmedBanditRouter."""

import pytest
import numpy as np
from models.routing.multi_armed_bandit import MultiArmedBanditRouter, MABRouteSelection


class TestMultiArmedBanditRouter:
    """Test suite for MultiArmedBanditRouter."""

    @pytest.fixture
    def router(self):
        return MultiArmedBanditRouter(n_routes=3, epsilon=0.1)

    @pytest.fixture
    def sample_data(self):
        X = np.random.rand(10, 4)
        y = np.random.randint(0, 3, 10)
        return X, y

    def test_initialization(self, router):
        """Test initialization."""
        assert router.n_routes == 3
        assert len(router.rewards) == 3
        assert len(router.counts) == 3
        assert not router.fitted

    def test_fit(self, router, sample_data):
        """Test training."""
        X, y = sample_data
        router.fit(X, y)

        assert router.fitted
        assert sum(router.counts) == 10
        assert any(r != 0.0 for r in router.rewards)

    def test_select_route_not_fitted(self, router):
        """Test route selection before training."""
        selected, scores = router.select_route()
        assert 0 <= selected < 3
        assert scores == {}

    def test_select_route_fitted(self, router, sample_data):
        """Test route selection after training."""
        X, y = sample_data
        router.fit(X, y)

        selected, scores = router.select_route()
        assert 0 <= selected < 3
        assert len(scores) == 3

    def test_update_reward(self, router):
        """Test reward update."""
        router.counts = [0, 0, 0]
        router.rewards = [0.0, 0.0, 0.0]

        router.update_reward(route=0, reward=10.0)
        assert router.counts[0] == 1
        assert router.rewards[0] == 10.0

        router.update_reward(route=0, reward=20.0)
        assert router.counts[0] == 2
        assert router.rewards[0] == 15.0  # Average of 10 and 20

    def test_predict_sample(self, router, sample_data):
        """Test prediction."""
        X, y = sample_data
        router.fit(X, y)

        prediction = router.predict_sample(X[0])
        assert isinstance(prediction, MABRouteSelection)
        assert 0 <= prediction.selected_route < 3
        assert len(prediction.ucb_scores) == 3
        assert len(prediction.arm_statistics) == 3

    def test_get_metrics(self, router, sample_data):
        """Test metrics retrieval."""
        X, y = sample_data
        router.fit(X, y)

        metrics = router.get_metrics()
        assert "n_routes" in metrics
        assert "total_pulls" in metrics
        assert metrics["total_pulls"] == 10

    def test_calculate_ucb(self, router):
        """Test UCB calculation."""
        router.counts = [10, 1, 0]
        router.rewards = [0.5, 0.8, 0.0]
        router.n_routes = 3

        ucb = router._calculate_ucb()
        assert len(ucb) == 3
        assert ucb[2] == float("inf")  # Unexplored arm
        assert ucb[1] > ucb[0]  # Less explored arm with higher reward
