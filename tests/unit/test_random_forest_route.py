"""Unit tests for RandomForestRouteClassifier."""

import numpy as np
import pytest

from models.routing.random_forest_route import RandomForestRouteClassifier, RouteSelection


class TestRandomForestRouteClassifier:
    """Test suite for RandomForestRouteClassifier."""

    @pytest.fixture
    def classifier(self):
        return RandomForestRouteClassifier(n_routes=3, n_trees=5)

    @pytest.fixture
    def sample_data(self):
        # Create simple separable data
        X = np.array(
            [
                [0.1, 0.1],
                [0.2, 0.2],  # Class 0
                [0.8, 0.8],
                [0.9, 0.9],  # Class 1
                [0.1, 0.9],
                [0.2, 0.8],  # Class 2
            ]
        )
        y = np.array([0, 0, 1, 1, 2, 2])
        return X, y

    def test_initialization(self, classifier):
        """Test initialization."""
        assert classifier.n_routes == 3
        assert classifier.n_trees == 5
        assert not classifier.fitted
        assert len(classifier.trees) == 0

    def test_fit(self, classifier, sample_data):
        """Test training."""
        X, y = sample_data
        classifier.fit(X, y)

        assert classifier.fitted
        assert len(classifier.trees) == 5

    def test_predict_not_fitted(self, classifier, sample_data):
        """Test prediction before training."""
        X, _ = sample_data
        with pytest.raises(RuntimeError):
            classifier.predict(X)

    def test_predict(self, classifier, sample_data):
        """Test prediction."""
        np.random.seed(42)
        X, y = sample_data
        classifier.fit(X, y)

        preds, confs = classifier.predict(X)
        assert len(preds) == len(X)
        assert len(confs) == len(X)
        # Allow some errors due to randomness/small data
        assert np.mean(preds == y) > 0.5

    def test_predict_sample(self, classifier, sample_data):
        """Test single sample prediction."""
        np.random.seed(42)
        X, y = sample_data
        classifier.fit(X, y)

        selection = classifier.predict_sample(X[0])
        assert isinstance(selection, RouteSelection)
        assert 0 <= selection.selected_route < 3
        assert len(selection.route_scores) == 3
        assert len(selection.top_3_routes) <= 3

    def test_score(self, classifier, sample_data):
        """Test scoring."""
        np.random.seed(42)
        X, y = sample_data
        classifier.fit(X, y)

        score = classifier.score(X, y)
        assert score > 0.5

    def test_get_metrics(self, classifier):
        """Test metrics retrieval."""
        metrics = classifier.get_metrics()
        assert "n_routes" in metrics
        assert "n_trees" in metrics
