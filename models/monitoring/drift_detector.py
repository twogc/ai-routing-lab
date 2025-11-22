"""Drift Detector - Detects data, concept, and performance drift in models."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class DriftType(Enum):
    """Types of drift"""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DRIFT = "performance_drift"


@dataclass
class DataDrift:
    """Data drift detection result"""

    is_drifted: bool
    drift_type: DriftType
    drift_score: float  # 0 to 1
    affected_features: List[str]
    statistical_test: str  # KL divergence, Wasserstein, etc.
    p_value: Optional[float] = None
    threshold: float = 0.05


@dataclass
class ConceptDrift:
    """Concept drift detection result"""

    is_drifted: bool
    performance_degradation: float  # Percentage
    baseline_accuracy: float
    current_accuracy: float
    window_size: int
    alert_threshold: float = 0.05  # 5% degradation


class DriftDetector:
    """
    Detects various types of drift in model data and performance.

    Detects:
    - Data Drift: Distribution changes in input features
    - Concept Drift: Relationship changes between inputs and outputs
    - Performance Drift: Model accuracy degradation
    """

    def __init__(
        self,
        window_size: int = 1000,
        drift_threshold: float = 0.05,
        logger: Optional[logging.Logger] = None,
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.logger = logger or logging.getLogger(__name__)

        self.baseline_distribution = None
        self.baseline_predictions = None
        self.baseline_accuracy = None
        self.fitted = False

    def fit(self, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> "DriftDetector":
        """
        Fit drift detector on baseline data.

        Args:
            X: Input features
            y_pred: Model predictions
            y_true: Ground truth labels

        Returns:
            Self for chaining
        """
        self.baseline_distribution = {
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0),
            "min": np.min(X, axis=0),
            "max": np.max(X, axis=0),
        }

        self.baseline_predictions = y_pred
        self.baseline_accuracy = np.mean(y_pred == y_true)

        self.fitted = True
        self.logger.info(
            f"DriftDetector fitted with baseline accuracy={self.baseline_accuracy:.4f}"
        )

        return self

    def detect_data_drift(self, X: np.ndarray, method: str = "kl_divergence") -> DataDrift:
        """
        Detect data drift using statistical tests.

        Args:
            X: Current input features
            method: Detection method (kl_divergence, wasserstein, kolmogorov_smirnov)

        Returns:
            DataDrift result
        """
        if not self.fitted:
            raise RuntimeError("DriftDetector must be fitted first")

        current_mean = np.mean(X, axis=0)
        current_std = np.std(X, axis=0)

        if method == "kl_divergence":
            drift_scores = self._kl_divergence(
                current_mean,
                current_std,
                self.baseline_distribution["mean"],
                self.baseline_distribution["std"],
            )
        elif method == "wasserstein":
            drift_scores = self._wasserstein_distance(
                current_mean,
                current_std,
                self.baseline_distribution["mean"],
                self.baseline_distribution["std"],
            )
        else:
            drift_scores = np.abs(current_mean - self.baseline_distribution["mean"])

        # Determine affected features
        affected_features = np.where(drift_scores > np.percentile(drift_scores, 75))[0]
        affected_names = [f"feature_{i}" for i in affected_features]

        # Overall drift score
        overall_drift = float(np.mean(drift_scores))
        is_drifted = overall_drift > self.drift_threshold

        return DataDrift(
            is_drifted=is_drifted,
            drift_type=DriftType.DATA_DRIFT,
            drift_score=overall_drift,
            affected_features=affected_names,
            statistical_test=method,
            p_value=self._calculate_p_value(drift_scores) if is_drifted else None,
            threshold=self.drift_threshold,
        )

    def detect_concept_drift(
        self, y_pred: np.ndarray, y_true: np.ndarray, window_size: Optional[int] = None
    ) -> ConceptDrift:
        """
        Detect concept drift via performance degradation.

        Args:
            y_pred: Current model predictions
            y_true: Ground truth labels
            window_size: Size of evaluation window

        Returns:
            ConceptDrift result
        """
        if self.baseline_accuracy is None:
            raise RuntimeError("DriftDetector must be fitted first")

        window_size = window_size or self.window_size
        current_accuracy = np.mean(y_pred[-window_size:] == y_true[-window_size:])
        degradation = self.baseline_accuracy - current_accuracy

        is_drifted = degradation > 0.05  # 5% threshold

        return ConceptDrift(
            is_drifted=is_drifted,
            performance_degradation=max(0, degradation * 100),
            baseline_accuracy=self.baseline_accuracy,
            current_accuracy=current_accuracy,
            window_size=window_size,
            alert_threshold=0.05,
        )

    def detect_performance_drift(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive performance drift detection.

        Returns dictionary with multiple metrics
        """
        accuracy = np.mean(y_pred == y_true)
        precision = self._calculate_precision(y_pred, y_true)
        recall = self._calculate_recall(y_pred, y_true)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": 2 * (precision * recall) / (precision + recall + 1e-10),
            "drift_detected": accuracy < self.baseline_accuracy * (1 - self.drift_threshold),
        }

    def get_drift_report(
        self, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, Any]:
        """Generate comprehensive drift report"""
        data_drift = self.detect_data_drift(X)
        concept_drift = self.detect_concept_drift(y_pred, y_true)
        perf_drift = self.detect_performance_drift(y_pred, y_true)

        overall_drifted = data_drift.is_drifted or concept_drift.is_drifted

        return {
            "overall_drift_detected": overall_drifted,
            "data_drift": {
                "detected": data_drift.is_drifted,
                "score": data_drift.drift_score,
                "affected_features": data_drift.affected_features,
            },
            "concept_drift": {
                "detected": concept_drift.is_drifted,
                "performance_degradation_percent": concept_drift.performance_degradation,
                "baseline_accuracy": concept_drift.baseline_accuracy,
                "current_accuracy": concept_drift.current_accuracy,
            },
            "performance_metrics": perf_drift,
        }

    @staticmethod
    def _kl_divergence(mean_p, std_p, mean_q, std_q) -> np.ndarray:
        """Calculate KL divergence between Gaussians"""
        var_p = std_p**2 + 1e-8
        var_q = std_q**2 + 1e-8

        kl = 0.5 * (np.log(var_q / var_p) + (var_p + (mean_p - mean_q) ** 2) / var_q - 1)

        return np.abs(kl)

    @staticmethod
    def _wasserstein_distance(mean_p, std_p, mean_q, std_q) -> np.ndarray:
        """Wasserstein distance between Gaussians"""
        return np.abs(mean_p - mean_q) + np.abs(std_p - std_q)

    @staticmethod
    def _calculate_p_value(scores: np.ndarray) -> float:
        """Simplified p-value calculation"""
        return float(np.mean(scores > np.percentile(scores, 95)))

    @staticmethod
    def _calculate_precision(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate precision"""
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return tp / (tp + fp + 1e-10)

    @staticmethod
    def _calculate_recall(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate recall"""
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return tp / (tp + fn + 1e-10)
