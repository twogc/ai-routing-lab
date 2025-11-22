"""Integration tests for quic-test collectors."""

from unittest.mock import Mock, patch

import pytest

from data.collectors.quic_test_collector import JSONFileCollector, PrometheusCollector


@pytest.mark.integration
class TestPrometheusCollector:
    """Integration tests for PrometheusCollector."""

    @patch("data.collectors.quic_test_collector.requests.Session")
    def test_collect_latency_metrics(self, mock_session, sample_prometheus_metrics):
        """Test collecting latency metrics from Prometheus."""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = sample_prometheus_metrics
        mock_response.raise_for_status = Mock()
        mock_session.return_value.get.return_value = mock_response

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        metrics = collector.collect_latency_metrics()

        assert "quic_latency_p95_ms" in metrics
        assert metrics["quic_latency_p95_ms"] == 25.5

    @patch("data.collectors.quic_test_collector.requests.Session")
    def test_collect_jitter_metrics(self, mock_session, sample_prometheus_metrics):
        """Test collecting jitter metrics from Prometheus."""
        mock_response = Mock()
        mock_response.json.return_value = sample_prometheus_metrics
        mock_response.raise_for_status = Mock()
        mock_session.return_value.get.return_value = mock_response

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        metrics = collector.collect_jitter_metrics()

        assert "quic_jitter_p95_ms" in metrics

    @patch("data.collectors.quic_test_collector.requests.Session")
    def test_collect_all_metrics(self, mock_session, sample_prometheus_metrics):
        """Test collecting all metrics."""
        mock_response = Mock()
        mock_response.json.return_value = sample_prometheus_metrics
        mock_response.raise_for_status = Mock()
        mock_session.return_value.get.return_value = mock_response

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        metrics = collector.collect_all_metrics()

        assert "latency" in metrics
        assert "jitter" in metrics
        assert "timestamp" in metrics

    @patch("data.collectors.quic_test_collector.requests.Session")
    def test_connection_error_handling(self, mock_session):
        """Test handling of connection errors."""
        mock_session.return_value.get.side_effect = Exception("Connection failed")

        collector = PrometheusCollector(prometheus_url="http://localhost:9090")
        metrics = collector.collect_latency_metrics()

        # Should return empty dict on error
        assert metrics == {}


@pytest.mark.integration
class TestJSONFileCollector:
    """Integration tests for JSONFileCollector."""

    def test_extract_metrics(self, sample_quic_test_json):
        """Test extracting metrics from quic-test JSON."""
        collector = JSONFileCollector(watch_directory="/tmp")
        metrics = collector.extract_metrics(sample_quic_test_json)

        assert "latency" in metrics
        assert metrics["latency"]["p95"] == 25.5
        assert metrics["latency"]["jitter"] == 2.3
        assert metrics["route_id"] == "route_0"

    def test_extract_metrics_with_throughput(self, sample_quic_test_json):
        """Test extracting throughput metrics."""
        collector = JSONFileCollector(watch_directory="/tmp")
        metrics = collector.extract_metrics(sample_quic_test_json)

        assert "throughput" in metrics
        assert metrics["throughput"]["average"] == 100.5

    def test_extract_metrics_missing_fields(self):
        """Test handling of missing fields in JSON."""
        collector = JSONFileCollector(watch_directory="/tmp")
        incomplete_data = {"RouteID": "test"}

        metrics = collector.extract_metrics(incomplete_data)

        # Should still have timestamp and route_id
        assert "timestamp" in metrics
        assert metrics["route_id"] == "test"
