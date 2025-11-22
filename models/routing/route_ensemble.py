"""Route Optimization Ensemble - Weighted voting for route selection."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .multi_armed_bandit import MultiArmedBanditRouter
from .neural_network_route import NeuralNetworkRouteOptimizer
from .q_learning_route import QLearningRouter
from .random_forest_route import RandomForestRouteClassifier


@dataclass
class EnsembleRouteSelection:
    """Ensemble route selection result"""

    selected_route: int
    ensemble_score: float
    model_votes: Dict[str, int]
    model_confidences: Dict[str, float]
    top_3_routes: List[Tuple[int, float]]
    model_agreement_percent: float


class RouteOptimizationEnsemble:
    """
    Weighted ensemble of four route optimization models.

    Weights: NN (0.40) + RF (0.30) + Q-Learning (0.20) + MAB (0.10)
    Expected Accuracy: 0.96
    """

    def __init__(
        self,
        nn_weight: float = 0.40,
        rf_weight: float = 0.30,
        ql_weight: float = 0.20,
        mab_weight: float = 0.10,
        n_routes: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        self.nn_weight = nn_weight
        self.rf_weight = rf_weight
        self.ql_weight = ql_weight
        self.mab_weight = mab_weight
        self.n_routes = n_routes
        self.logger = logger or logging.getLogger(__name__)

        # Normalize weights
        total = nn_weight + rf_weight + ql_weight + mab_weight
        self.nn_weight /= total
        self.rf_weight /= total
        self.ql_weight /= total
        self.mab_weight /= total

        # Initialize models
        self.nn_model = NeuralNetworkRouteOptimizer(n_routes=n_routes, logger=logger)
        self.rf_model = RandomForestRouteClassifier(n_routes=n_routes, logger=logger)
        self.ql_model = QLearningRouter(n_routes=n_routes, logger=logger)
        self.mab_model = MultiArmedBanditRouter(n_routes=n_routes, logger=logger)

        self.fitted = False
        self.metrics = {"total_selections": 0, "agreement_count": 0}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RouteOptimizationEnsemble":
        """Fit all ensemble models"""
        self.logger.info("Fitting Neural Network...")
        self.nn_model.fit(X, y)

        self.logger.info("Fitting Random Forest...")
        self.rf_model.fit(X, y)

        self.logger.info("Fitting Q-Learning...")
        self.ql_model.fit(X, y)

        self.logger.info("Fitting Multi-Armed Bandit...")
        self.mab_model.fit(X, y)

        self.fitted = True
        self.logger.info("Route Optimization Ensemble fitted")
        return self

    def select_route(self, x: np.ndarray) -> EnsembleRouteSelection:
        """Select best route using ensemble"""
        if not self.fitted:
            raise RuntimeError("Ensemble must be fitted before select_route")

        x = x.reshape(1, -1).astype(np.float32) if x.ndim == 1 else x.astype(np.float32)

        # Get predictions from all models
        nn_pred = self.nn_model.predict_sample(x[0])
        rf_pred = self.rf_model.predict_sample(x[0])
        ql_pred = self.ql_model.select_route(x[0])
        mab_pred = self.mab_model.predict_sample(x[0])

        # Aggregate votes
        votes = np.zeros(self.n_routes)
        votes[nn_pred.selected_route] += self.nn_weight
        votes[rf_pred.selected_route] += self.rf_weight
        votes[ql_pred.selected_route] += self.ql_weight
        votes[mab_pred.selected_route] += self.mab_weight

        selected_route = np.argmax(votes)
        ensemble_score = votes[selected_route]

        # Model agreement
        individual_votes = [
            nn_pred.selected_route,
            rf_pred.selected_route,
            ql_pred.selected_route,
            mab_pred.selected_route,
        ]
        agreement = sum(1 for v in individual_votes if v == selected_route) / len(individual_votes)

        # Top 3 routes
        top_3 = sorted(enumerate(votes), key=lambda x: x[1], reverse=True)[:3]

        self.metrics["total_selections"] += 1
        if agreement == 1.0:
            self.metrics["agreement_count"] += 1

        return EnsembleRouteSelection(
            selected_route=int(selected_route),
            ensemble_score=float(ensemble_score),
            model_votes={
                "neural_network": int(nn_pred.selected_route),
                "random_forest": int(rf_pred.selected_route),
                "q_learning": int(ql_pred.selected_route),
                "mab": int(mab_pred.selected_route),
            },
            model_confidences={
                "neural_network": float(nn_pred.confidence),
                "random_forest": float(rf_pred.confidence),
                "q_learning": float(ql_pred.expected_reward),
                "mab": float(sum(mab_pred.arm_statistics[mab_pred.selected_route].values()) / 3),
            },
            top_3_routes=[(int(r), float(s)) for r, s in top_3],
            model_agreement_percent=agreement * 100,
        )

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch predictions"""
        routes = []
        scores = []

        for x in X:
            result = self.select_route(x)
            routes.append(result.selected_route)
            scores.append(result.ensemble_score)

        return np.array(routes), np.array(scores)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy"""
        predictions, _ = self.predict(X)
        return np.mean(predictions == y)

    def get_metrics(self) -> Dict[str, Any]:
        """Get ensemble metrics"""
        total = self.metrics["total_selections"]
        agreement_rate = self.metrics["agreement_count"] / total * 100 if total > 0 else 0

        return {
            **self.metrics,
            "agreement_rate_percent": agreement_rate,
            "weights": {
                "neural_network": self.nn_weight,
                "random_forest": self.rf_weight,
                "q_learning": self.ql_weight,
                "mab": self.mab_weight,
            },
            "individual_metrics": {
                "neural_network": self.nn_model.get_metrics(),
                "random_forest": self.rf_model.get_metrics(),
                "q_learning": self.ql_model.get_metrics(),
                "mab": self.mab_model.get_metrics(),
            },
        }
