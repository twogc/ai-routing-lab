"""Unit tests for DriftDetector."""

import pytest
import numpy as np
from models.monitoring.drift_detector import DriftDetector, DataDrift, ConceptDrift


class TestDriftDetector:
    """Test suite for DriftDetector."""

    @pytest.fixture
    def detector(self):
        return DriftDetector(window_size=10)

    @pytest.fixture
    def baseline_data(self):
        X = np.random.normal(0, 1, (100, 5))
        y_true = np.random.randint(0, 2, 100)
        y_pred = y_true.copy()  # Perfect prediction
        return X, y_pred, y_true

    def test_initialization(self, detector):
        """Test initialization."""
        assert detector.window_size == 10
        assert not detector.fitted
        assert detector.baseline_accuracy is None

    def test_fit(self, detector, baseline_data):
        """Test fitting."""
        X, y_pred, y_true = baseline_data
        detector.fit(X, y_pred, y_true)

        assert detector.fitted
        assert detector.baseline_accuracy == 1.0
        assert detector.baseline_distribution is not None

    def test_detect_data_drift_no_drift(self, detector, baseline_data):
        """Test data drift detection (no drift)."""
        X, y_pred, y_true = baseline_data
        detector.fit(X, y_pred, y_true)

        # Same distribution
        X_new = np.random.normal(0, 1, (50, 5))
        drift = detector.detect_data_drift(X_new)

        assert not drift.is_drifted
        assert drift.drift_score < detector.drift_threshold

    def test_detect_data_drift_with_drift(self, detector, baseline_data):
        """Test data drift detection (with drift)."""
        X, y_pred, y_true = baseline_data
        detector.fit(X, y_pred, y_true)

        # Different distribution
        X_new = np.random.normal(5, 1, (50, 5))
        drift = detector.detect_data_drift(X_new)

        assert drift.is_drifted
        assert drift.drift_score > detector.drift_threshold
        assert len(drift.affected_features) > 0

    def test_detect_concept_drift(self, detector, baseline_data):
        """Test concept drift detection."""
        X, y_pred, y_true = baseline_data
        detector.fit(X, y_pred, y_true)

        # Degraded performance
        y_pred_new = np.zeros(10)
        y_true_new = np.ones(10)

        drift = detector.detect_concept_drift(y_pred_new, y_true_new)

        assert drift.is_drifted
        assert drift.performance_degradation > 0
        assert drift.current_accuracy == 0.0

    def test_detect_performance_drift(self, detector, baseline_data):
        """Test performance drift detection."""
        X, y_pred, y_true = baseline_data
        detector.fit(X, y_pred, y_true)

        y_pred_new = np.zeros(10)
        y_true_new = np.ones(10)

        metrics = detector.detect_performance_drift(y_pred_new, y_true_new)

        assert metrics["accuracy"] == 0.0
        assert metrics["drift_detected"]

    def test_get_drift_report(self, detector, baseline_data):
        """Test drift report generation."""
        X, y_pred, y_true = baseline_data
        detector.fit(X, y_pred, y_true)

        report = detector.get_drift_report(X, y_pred, y_true)

        assert "overall_drift_detected" in report
        assert "data_drift" in report
        assert "concept_drift" in report
        assert "performance_metrics" in report

    def test_methods(self, detector, baseline_data):
        """Test different drift detection methods."""
        X, y_pred, y_true = baseline_data
        detector.fit(X, y_pred, y_true)
        X_new = np.random.normal(0, 1, (50, 5))

        drift_kl = detector.detect_data_drift(X_new, method="kl_divergence")
        drift_ws = detector.detect_data_drift(X_new, method="wasserstein")
        drift_simple = detector.detect_data_drift(X_new, method="simple")

        assert isinstance(drift_kl, DataDrift)
        assert isinstance(drift_ws, DataDrift)
        assert isinstance(drift_simple, DataDrift)
