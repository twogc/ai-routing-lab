"""Unit tests for ModelMonitor."""

import pytest

from models.monitoring.model_monitor import ModelHealth, ModelMonitor


class TestModelMonitor:
    """Test suite for ModelMonitor."""

    @pytest.fixture
    def monitor(self):
        return ModelMonitor(model_id="test_model")

    def test_initialization(self, monitor):
        """Test initialization."""
        assert monitor.model_id == "test_model"
        assert monitor.predictions_total == 0
        assert len(monitor.latencies) == 0

    def test_record_prediction(self, monitor):
        """Test recording predictions."""
        monitor.record_prediction(prediction=1, ground_truth=1, latency_ms=10.0)
        monitor.record_prediction(prediction=1, ground_truth=0, latency_ms=20.0)

        assert monitor.predictions_total == 2
        assert monitor.predictions_correct == 1
        assert len(monitor.latencies) == 2

    def test_get_current_metrics(self, monitor):
        """Test current metrics calculation."""
        monitor.record_prediction(prediction=1, ground_truth=1, latency_ms=10.0)
        monitor.record_prediction(prediction=1, ground_truth=0, latency_ms=20.0)

        metrics = monitor.get_current_metrics()
        assert isinstance(metrics, ModelHealth)
        assert metrics.accuracy == 0.5
        assert metrics.latency_p50_ms == 15.0
        assert metrics.status == "degraded"  # accuracy 0.5 < 0.85

    def test_get_metrics_summary(self, monitor):
        """Test metrics summary."""
        monitor.record_prediction(prediction=1, ground_truth=1, latency_ms=10.0)
        monitor.get_current_metrics()

        summary = monitor.get_metrics_summary()
        assert summary["model_id"] == "test_model"
        assert summary["metrics_count"] == 1
        assert summary["accuracy"]["mean"] == 1.0

    def test_get_prometheus_metrics(self, monitor):
        """Test Prometheus metrics format."""
        monitor.record_prediction(prediction=1, ground_truth=1)

        output = monitor.get_prometheus_metrics()
        assert "cloudbridge_model_accuracy" in output
        assert 'model_id="test_model"' in output

    def test_reset_metrics(self, monitor):
        """Test resetting metrics."""
        monitor.record_prediction(prediction=1, ground_truth=1)
        monitor.reset_metrics()

        assert monitor.predictions_total == 0
        assert len(monitor.latencies) == 0

    def test_health_status(self, monitor):
        """Test health status determination."""
        # Healthy
        monitor.record_prediction(prediction=1, ground_truth=1, latency_ms=10.0)
        metrics = monitor.get_current_metrics()
        assert metrics.status == "healthy"

        # Degraded (low accuracy)
        monitor.reset_metrics()
        monitor.record_prediction(prediction=0, ground_truth=1, latency_ms=10.0)
        metrics = monitor.get_current_metrics()
        assert metrics.status == "degraded"

        # Unhealthy (low accuracy + high latency)
        monitor.reset_metrics()
        monitor.record_prediction(prediction=0, ground_truth=1, latency_ms=200.0)
        metrics = monitor.get_current_metrics()
        assert metrics.status == "unhealthy"
