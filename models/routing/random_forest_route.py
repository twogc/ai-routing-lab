"""Random Forest Route Classifier - Multi-class route selection."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RouteSelection:
    """Route selection result"""

    selected_route: int
    route_scores: Dict[int, float]
    confidence: float
    top_3_routes: List[Tuple[int, float]]


class RandomForestRouteClassifier:
    """Multi-class route classifier using random forest."""

    def __init__(
        self, n_routes: int = 5, n_trees: int = 50, logger: Optional[logging.Logger] = None
    ):
        self.n_routes = n_routes
        self.n_trees = n_trees
        self.logger = logger or logging.getLogger(__name__)
        self.trees = []
        self.fitted = False
        self.metrics = {"accuracy": 0.0, "total_predictions": 0, "correct_predictions": 0}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRouteClassifier":
        """Fit random forest classifier"""
        X = X.astype(np.float32)
        y = y.astype(int)

        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            tree = self._build_tree(X_boot, y_boot, depth=0)
            self.trees.append(tree)

        self.fitted = True
        self.logger.info(f"RandomForest route classifier fitted with {self.n_trees} trees")
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict routes"""
        if not self.fitted:
            raise RuntimeError("Model must be fitted first")

        predictions = np.zeros(len(X), dtype=int)
        confidences = np.zeros(len(X))

        for i, x in enumerate(X):
            votes = np.zeros(self.n_routes)
            for tree in self.trees:
                route = self._predict_sample(x, tree)
                votes[route] += 1

            predictions[i] = np.argmax(votes)
            confidences[i] = votes[predictions[i]] / len(self.trees)
            self.metrics["total_predictions"] += 1

        return predictions, confidences

    def predict_sample(self, x: np.ndarray) -> RouteSelection:
        """Predict route for single sample"""
        x = x.reshape(1, -1)
        preds, confs = self.predict(x)

        votes = np.zeros(self.n_routes)
        for tree in self.trees:
            route = self._predict_sample(x[0], tree)
            votes[route] += 1

        route_scores = {i: votes[i] / len(self.trees) for i in range(self.n_routes)}
        top_3 = sorted(route_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        return RouteSelection(
            selected_route=int(preds[0]),
            route_scores=route_scores,
            confidence=float(confs[0]),
            top_3_routes=top_3,
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy"""
        preds, _ = self.predict(X)
        return np.mean(preds == y)

    def get_metrics(self) -> Dict[str, Any]:
        return {**self.metrics, "n_routes": self.n_routes, "n_trees": self.n_trees}

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int, max_depth: int = 10) -> Dict:
        if depth >= max_depth or len(np.unique(y)) == 1:
            return {"leaf": True, "class": np.argmax(np.bincount(y, minlength=self.n_routes))}

        best_score = float("inf")
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                gini = self._gini(y[left_mask], y[right_mask])
                if gini < best_score:
                    best_score, best_feature, best_threshold = gini, feature, threshold

        if best_feature is None:
            return {"leaf": True, "class": np.argmax(np.bincount(y, minlength=self.n_routes))}

        left_mask = X[:, best_feature] < best_threshold
        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1, max_depth),
            "right": self._build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth),
        }

    def _predict_sample(self, x: np.ndarray, tree: Dict) -> int:
        if tree["leaf"]:
            return tree["class"]
        return self._predict_sample(
            x, tree["left" if x[tree["feature"]] < tree["threshold"] else "right"]
        )

    @staticmethod
    def _gini(y_left: np.ndarray, y_right: np.ndarray) -> float:
        def gini_index(y):
            classes, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return 1 - np.sum(probs**2)

        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right
        return (n_left / n_total) * gini_index(y_left) + (n_right / n_total) * gini_index(y_right)
