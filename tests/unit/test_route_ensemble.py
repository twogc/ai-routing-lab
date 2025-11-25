"""Unit tests for RouteOptimizationEnsemble."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models.routing.route_ensemble import EnsembleRouteSelection, RouteOptimizationEnsemble


class TestRouteOptimizationEnsemble:
    """Test suite for RouteOptimizationEnsemble."""

    @pytest.fixture
    def mock_nn(self):
        mock = MagicMock()
        mock.predict_sample.return_value = MagicMock(selected_route=0, confidence=0.9)
        mock.get_metrics.return_value = {}
        return mock

    @pytest.fixture
    def mock_rf(self):
        mock = MagicMock()
        mock.predict_sample.return_value = MagicMock(selected_route=0, confidence=0.8)
        mock.get_metrics.return_value = {}
        return mock

    @pytest.fixture
    def mock_ql(self):
        mock = MagicMock()
        mock.select_route.return_value = MagicMock(selected_route=1, expected_reward=10.0)
        mock.get_metrics.return_value = {}
        return mock

    @pytest.fixture
    def mock_mab(self):
        mock = MagicMock()
        mock.predict_sample.return_value = MagicMock(
            selected_route=1, arm_statistics={1: {"mean": 0.5, "count": 10, "ucb": 0.6}}
        )
        mock.get_metrics.return_value = {}
        return mock

    @pytest.fixture
    def ensemble(self, mock_nn, mock_rf, mock_ql, mock_mab):
        with (
            patch(
                "models.routing.route_ensemble.NeuralNetworkRouteOptimizer", return_value=mock_nn
            ),
            patch(
                "models.routing.route_ensemble.RandomForestRouteClassifier", return_value=mock_rf
            ),
            patch("models.routing.route_ensemble.QLearningRouter", return_value=mock_ql),
            patch("models.routing.route_ensemble.MultiArmedBanditRouter", return_value=mock_mab),
        ):

            ensemble = RouteOptimizationEnsemble(n_routes=3)
            # Manually set mocks because __init__ creates new instances
            ensemble.nn_model = mock_nn
            ensemble.rf_model = mock_rf
            ensemble.ql_model = mock_ql
            ensemble.mab_model = mock_mab
            return ensemble

    def test_initialization(self, ensemble):
        """Test initialization."""
        assert ensemble.n_routes == 3
        assert not ensemble.fitted
        # Check normalized weights
        total = ensemble.nn_weight + ensemble.rf_weight + ensemble.ql_weight + ensemble.mab_weight
        assert abs(total - 1.0) < 1e-6

    def test_fit(self, ensemble):
        """Test training."""
        X = np.zeros((5, 4))
        y = np.zeros(5)

        ensemble.fit(X, y)

        assert ensemble.fitted
        ensemble.nn_model.fit.assert_called_once()
        ensemble.rf_model.fit.assert_called_once()
        ensemble.ql_model.fit.assert_called_once()
        ensemble.mab_model.fit.assert_called_once()

    def test_select_route_not_fitted(self, ensemble):
        """Test prediction before training."""
        with pytest.raises(RuntimeError):
            ensemble.select_route(np.zeros(4))

    def test_select_route(self, ensemble):
        """Test route selection logic."""
        ensemble.fitted = True
        X = np.zeros(4)

        # NN(0.4)*0 + RF(0.3)*0 + QL(0.2)*1 + MAB(0.1)*1
        # Route 0: 0.7 votes
        # Route 1: 0.3 votes

        selection = ensemble.select_route(X)

        assert isinstance(selection, EnsembleRouteSelection)
        assert selection.selected_route == 0
        assert selection.model_votes["neural_network"] == 0
        assert selection.model_votes["q_learning"] == 1

    def test_predict(self, ensemble):
        """Test batch prediction."""
        ensemble.fitted = True
        X = np.zeros((2, 4))

        routes, scores = ensemble.predict(X)

        assert len(routes) == 2
        assert len(scores) == 2
        assert routes[0] == 0

    def test_score(self, ensemble):
        """Test scoring."""
        ensemble.fitted = True
        X = np.zeros((2, 4))
        y = np.array([0, 0])

        score = ensemble.score(X, y)
        assert score == 1.0

    def test_get_metrics(self, ensemble):
        """Test metrics retrieval."""
        ensemble.fitted = True
        ensemble.select_route(np.zeros(4))

        metrics = ensemble.get_metrics()
        assert "agreement_rate_percent" in metrics
        assert "weights" in metrics
        assert "individual_metrics" in metrics
