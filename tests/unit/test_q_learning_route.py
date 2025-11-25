"""Unit tests for QLearningRouter."""

import pytest
import numpy as np
from models.routing.q_learning_route import QLearningRouter, QLearningDecision


class TestQLearningRouter:
    """Test suite for QLearningRouter."""

    @pytest.fixture
    def router(self):
        return QLearningRouter(n_routes=3, n_states=5)

    @pytest.fixture
    def sample_data(self):
        X = np.random.rand(10, 4)
        y = np.random.randint(0, 3, 10)
        return X, y

    def test_initialization(self, router):
        """Test initialization."""
        assert router.n_routes == 3
        assert router.n_states == 5
        assert router.q_table.shape == (5, 3)
        assert not router.fitted

    def test_fit(self, router, sample_data):
        """Test training."""
        X, y = sample_data
        router.fit(X, y, epochs=2)

        assert router.fitted
        assert router.total_updates > 0
        assert np.any(router.q_table != 0)

    def test_select_route_not_fitted(self, router, sample_data):
        """Test route selection before training."""
        X, _ = sample_data
        decision = router.select_route(X[0])

        assert decision.selected_route == 0
        assert all(v == 0.0 for v in decision.q_values.values())

    def test_select_route_fitted(self, router, sample_data):
        """Test route selection after training."""
        X, y = sample_data
        router.fit(X, y, epochs=5)

        decision = router.select_route(X[0])

        assert isinstance(decision, QLearningDecision)
        assert 0 <= decision.selected_route < 3
        assert len(decision.q_values) == 3

    def test_select_route_explore(self, router, sample_data):
        """Test exploration mode."""
        X, y = sample_data
        router.fit(X, y)

        # Force exploration (mock random to ensure explore path taken?
        # Or just run multiple times and check valid output)
        decision = router.select_route(X[0], explore=True)
        assert 0 <= decision.selected_route < 3

    def test_update(self, router, sample_data):
        """Test single update."""
        X, _ = sample_data
        initial_q = router.q_table.copy()

        router.update(X[0], action=1, reward=10.0)

        assert router.total_updates == 1
        assert not np.array_equal(router.q_table, initial_q)

    def test_get_metrics(self, router, sample_data):
        """Test metrics retrieval."""
        X, y = sample_data
        router.fit(X, y)

        metrics = router.get_metrics()
        assert "n_routes" in metrics
        assert "mean_q_value" in metrics
        assert metrics["total_updates"] > 0

    def test_discretize_state(self, router):
        """Test state discretization."""
        x = np.array([0.5, 0.5])
        state = router._discretize_state(x)
        assert isinstance(state, int)
        assert 0 <= state < router.n_states
