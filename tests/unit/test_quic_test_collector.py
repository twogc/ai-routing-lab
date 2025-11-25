"""Unit tests for QuicTestCollector."""

import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from data.collectors.quic_test_collector import JSONFileCollector, PrometheusCollector


@pytest.fixture
def mock_response():
    """Mock requests response."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"__name__": "quic_latency_p95_ms", "route_id": "route_0"},
                    "value": [1234567890, "25.5"],
                }
            ]
        },
    }
    return mock_resp


class TestPrometheusCollector:
    """Test suite for PrometheusCollector."""

    @patch("requests.Session.get")
    def test_collect_latency_metrics(self, mock_get, mock_response):
        """Test collecting latency metrics."""
        mock_get.return_value = mock_response

        collector = PrometheusCollector()
        metrics = collector.collect_latency_metrics()

        assert "quic_latency_p95_ms" in metrics
        assert metrics["quic_latency_p95_ms"] == 25.5

        # Verify query param
        args, kwargs = mock_get.call_args
        assert "query" in kwargs["params"]
        assert "quic_latency_p95_ms" in kwargs["params"]["query"]

    @patch("requests.Session.get")
    def test_collect_latency_metrics_with_route(self, mock_get, mock_response):
        """Test collecting latency metrics with route filter."""
        mock_get.return_value = mock_response

        collector = PrometheusCollector()
        metrics = collector.collect_latency_metrics(route_id="route_0")

        # Verify query param includes route_id
        args, kwargs = mock_get.call_args
        assert 'route_id="route_0"' in kwargs["params"]["query"]

    @patch("requests.Session.get")
    def test_collect_metrics_error(self, mock_get):
        """Test error handling during collection."""
        mock_get.side_effect = Exception("Connection error")

        collector = PrometheusCollector()
        metrics = collector.collect_latency_metrics()

        assert metrics == {}

    @patch("requests.Session.get")
    def test_collect_jitter_metrics(self, mock_get, mock_response):
        """Test collecting jitter metrics."""
        mock_response.json.return_value["data"]["result"][0]["metric"][
            "__name__"
        ] = "quic_jitter_p95_ms"
        mock_get.return_value = mock_response

        collector = PrometheusCollector()
        metrics = collector.collect_jitter_metrics()

        assert "quic_jitter_p95_ms" in metrics
        assert metrics["quic_jitter_p95_ms"] == 25.5

    @patch("requests.Session.get")
    def test_collect_all_metrics(self, mock_get, mock_response):
        """Test collecting all metrics."""
        mock_get.return_value = mock_response

        collector = PrometheusCollector()
        metrics = collector.collect_all_metrics()

        assert "latency" in metrics
        assert "jitter" in metrics
        assert "timestamp" in metrics


class TestJSONFileCollector:
    """Test suite for JSONFileCollector."""

    @pytest.fixture
    def sample_json_data(self):
        return {
            "RouteID": "route_0",
            "Latency": {"P50": 20.0, "P95": 25.5, "Jitter": 2.3},
            "Throughput": {"Average": 100.5},
        }

    def test_extract_metrics(self, sample_json_data):
        """Test extracting metrics from JSON data."""
        collector = JSONFileCollector(watch_directory="/tmp")
        metrics = collector.extract_metrics(sample_json_data)

        assert metrics["route_id"] == "route_0"
        assert metrics["latency"]["p95"] == 25.5
        assert metrics["latency"]["jitter"] == 2.3
        assert metrics["throughput"]["average"] == 100.5
        assert "timestamp" in metrics

    def test_process_file(self, sample_json_data):
        """Test processing a JSON file."""
        mock_callback = MagicMock()
        collector = JSONFileCollector(watch_directory="/tmp", output_callback=mock_callback)

        with patch("builtins.open", mock_open(read_data=json.dumps(sample_json_data))):
            collector.process_file("/tmp/test.json")

        mock_callback.assert_called_once()
        args, _ = mock_callback.call_args
        metrics = args[0]
        assert metrics["route_id"] == "route_0"

    def test_on_created(self):
        """Test file creation event handler."""
        collector = JSONFileCollector(watch_directory="/tmp")
        collector.process_file = MagicMock()

        mock_event = MagicMock()
        mock_event.src_path = "/tmp/test.json"

        collector.on_created(mock_event)
        collector.process_file.assert_called_with("/tmp/test.json")

        # Should ignore non-json files
        mock_event.src_path = "/tmp/test.txt"
        collector.process_file.reset_mock()
        collector.on_created(mock_event)
        collector.process_file.assert_not_called()
