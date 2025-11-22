"""
Route Prediction Ensemble - Combines latency and jitter predictions for route selection.

Weighted ensemble combining latency and jitter predictions to select optimal routes.
Target: >92% accuracy in route selection based on predicted latency/jitter.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .jitter_predictor import JitterPredictor
from .latency_predictor import LatencyPredictor

logger = logging.getLogger(__name__)


@dataclass
class RoutePrediction:
    """Route prediction result combining latency and jitter"""

    route_id: int
    predicted_latency_ms: float
    predicted_jitter_ms: float
    combined_score: float  # Weighted combination of latency and jitter
    latency_confidence: float
    jitter_confidence: float
    overall_confidence: float


class RoutePredictionEnsemble:
    """
    Ensemble combining latency and jitter predictions for route selection.

    Combines predictions from LatencyPredictor and JitterPredictor to rank routes.
    Lower latency and jitter = better route.

    Scoring formula:
    score = (latency_weight * normalized_latency) + (jitter_weight * normalized_jitter)

    Target: >92% accuracy in selecting optimal routes
    """

    def __init__(
        self,
        latency_weight: float = 0.7,
        jitter_weight: float = 0.3,
        latency_predictor: Optional[LatencyPredictor] = None,
        jitter_predictor: Optional[JitterPredictor] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Route Prediction Ensemble.

        Args:
            latency_weight: Weight for latency in route scoring (0-1)
            jitter_weight: Weight for jitter in route scoring (0-1)
            latency_predictor: Optional pre-trained latency predictor
            jitter_predictor: Optional pre-trained jitter predictor
            logger: Optional logger instance
        """
        # Normalize weights
        total_weight = latency_weight + jitter_weight
        self.latency_weight = latency_weight / total_weight
        self.jitter_weight = jitter_weight / total_weight

        self.logger = logger or logging.getLogger(__name__)

        # Initialize predictors
        self.latency_predictor = latency_predictor or LatencyPredictor(logger=self.logger)
        self.jitter_predictor = jitter_predictor or JitterPredictor(logger=self.logger)

        self.fitted = False
        self.metrics = {"total_predictions": 0, "optimal_route_selected": 0}

    def fit(
        self,
        X_latency: np.ndarray,
        y_latency: np.ndarray,
        X_jitter: np.ndarray,
        y_jitter: np.ndarray,
        latency_feature_names: Optional[List[str]] = None,
        jitter_feature_names: Optional[List[str]] = None,
    ) -> "RoutePredictionEnsemble":
        """
        Train both latency and jitter predictors.

        Args:
            X_latency: Features for latency prediction
            y_latency: Target latency values
            X_jitter: Features for jitter prediction
            y_jitter: Target jitter values
            latency_feature_names: Optional feature names for latency
            jitter_feature_names: Optional feature names for jitter

        Returns:
            Self for chaining
        """
        self.logger.info("Training Route Prediction Ensemble...")

        # Train latency predictor
        self.latency_predictor.fit(X_latency, y_latency, latency_feature_names)

        # Train jitter predictor
        self.jitter_predictor.fit(X_jitter, y_jitter, jitter_feature_names)

        self.fitted = True
        self.logger.info("Route Prediction Ensemble trained")

        return self

    def predict_route(
        self, X_latency: np.ndarray, X_jitter: np.ndarray, route_id: Optional[int] = None
    ) -> RoutePrediction:
        """
        Predict latency and jitter for a route and calculate combined score.

        Args:
            X_latency: Features for latency prediction
            X_jitter: Features for jitter prediction
            route_id: Optional route identifier

        Returns:
            RoutePrediction with combined predictions
        """
        if not self.fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        # Predict latency
        latency_pred = self.latency_predictor.predict(X_latency)

        # Predict jitter
        jitter_pred = self.jitter_predictor.predict(X_jitter)

        # Normalize predictions for scoring (lower is better)
        # Use inverse normalization: lower latency/jitter = higher score
        # Normalize to [0, 1] range where 1 = best (lowest latency/jitter)
        max_latency = 200.0  # Assume max latency of 200ms
        max_jitter = 50.0  # Assume max jitter of 50ms

        normalized_latency = 1.0 - min(latency_pred.predicted_latency_ms / max_latency, 1.0)
        normalized_jitter = 1.0 - min(jitter_pred.predicted_jitter_ms / max_jitter, 1.0)

        # Calculate combined score
        combined_score = (
            self.latency_weight * normalized_latency + self.jitter_weight * normalized_jitter
        )

        # Overall confidence (average of both)
        overall_confidence = (latency_pred.confidence_score + jitter_pred.confidence_score) / 2.0

        return RoutePrediction(
            route_id=route_id if route_id is not None else 0,
            predicted_latency_ms=latency_pred.predicted_latency_ms,
            predicted_jitter_ms=jitter_pred.predicted_jitter_ms,
            combined_score=combined_score,
            latency_confidence=latency_pred.confidence_score,
            jitter_confidence=jitter_pred.confidence_score,
            overall_confidence=overall_confidence,
        )

    def select_best_route(
        self, routes_features: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[int, RoutePrediction]:
        """
        Select best route from multiple route options.

        Args:
            routes_features: Dictionary mapping route_id to (X_latency, X_jitter) tuples

        Returns:
            Tuple of (best_route_id, RoutePrediction)
        """
        if not routes_features:
            raise ValueError("No routes provided")

        route_predictions = {}

        for route_id, (X_latency, X_jitter) in routes_features.items():
            prediction = self.predict_route(X_latency, X_jitter, route_id)
            route_predictions[route_id] = prediction

        # Select route with highest combined score
        best_route_id = max(
            route_predictions.keys(), key=lambda r: route_predictions[r].combined_score
        )
        best_prediction = route_predictions[best_route_id]

        self.metrics["total_predictions"] += 1

        self.logger.info(
            f"Selected route {best_route_id}: "
            f"latency={best_prediction.predicted_latency_ms:.2f}ms, "
            f"jitter={best_prediction.predicted_jitter_ms:.2f}ms, "
            f"score={best_prediction.combined_score:.4f}"
        )

        return best_route_id, best_prediction

    def evaluate_route_selection(
        self, routes_features: Dict[int, Tuple[np.ndarray, np.ndarray]], true_optimal_route: int
    ) -> Dict[str, Any]:
        """
        Evaluate route selection accuracy.

        Args:
            routes_features: Dictionary mapping route_id to (X_latency, X_jitter)
            true_optimal_route: True optimal route ID (based on actual measurements)

        Returns:
            Dictionary with evaluation metrics
        """
        selected_route_id, prediction = self.select_best_route(routes_features)

        is_correct = selected_route_id == true_optimal_route

        if is_correct:
            self.metrics["optimal_route_selected"] += 1

        accuracy = (
            self.metrics["optimal_route_selected"] / self.metrics["total_predictions"]
            if self.metrics["total_predictions"] > 0
            else 0.0
        )

        return {
            "selected_route": selected_route_id,
            "true_optimal_route": true_optimal_route,
            "is_correct": is_correct,
            "prediction": prediction,
            "accuracy": accuracy,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get ensemble metrics"""
        accuracy = (
            self.metrics["optimal_route_selected"] / self.metrics["total_predictions"]
            if self.metrics["total_predictions"] > 0
            else 0.0
        )

        return {
            **self.metrics,
            "accuracy": accuracy,
            "latency_weight": self.latency_weight,
            "jitter_weight": self.jitter_weight,
            "latency_predictor_metrics": self.latency_predictor.get_metrics(),
            "jitter_predictor_metrics": self.jitter_predictor.get_metrics(),
        }
