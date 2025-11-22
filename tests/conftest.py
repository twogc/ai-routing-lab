"""Pytest configuration and fixtures."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_features():
    """Sample feature data for testing."""
    return np.array(
        [
            [25.5, 2.3, 0.95, 1.0],
            [30.1, 3.1, 0.85, 1.2],
            [20.3, 1.8, 0.98, 0.8],
            [35.2, 4.2, 0.75, 1.5],
            [22.1, 2.1, 0.92, 0.9],
        ]
    )


@pytest.fixture
def sample_latency_targets():
    """Sample latency target values."""
    return np.array([26.0, 31.0, 21.0, 36.0, 23.0])


@pytest.fixture
def sample_jitter_targets():
    """Sample jitter target values."""
    return np.array([2.5, 3.2, 1.9, 4.3, 2.2])


@pytest.fixture
def temp_models_dir():
    """Temporary directory for model storage."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_route_data():
    """Sample route data for testing."""
    return {
        "route_0": {"latency_features": [25.5, 2.3], "jitter_features": [2.3, 0.5]},
        "route_1": {"latency_features": [30.1, 3.1], "jitter_features": [3.1, 0.8]},
        "route_2": {"latency_features": [20.3, 1.8], "jitter_features": [1.8, 0.3]},
    }


@pytest.fixture
def sample_prometheus_metrics():
    """Sample Prometheus metrics response."""
    return {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"__name__": "quic_latency_p95_ms", "route_id": "route_0"},
                    "value": [1234567890, "25.5"],
                },
                {
                    "metric": {"__name__": "quic_jitter_p95_ms", "route_id": "route_0"},
                    "value": [1234567890, "2.3"],
                },
            ]
        },
    }


@pytest.fixture
def sample_quic_test_json():
    """Sample quic-test JSON export."""
    return {
        "RouteID": "route_0",
        "SourcePoP": "pop_1",
        "TargetPoP": "pop_2",
        "Latency": {"P50": 20.0, "P95": 25.5, "P99": 30.0, "Average": 22.5, "Jitter": 2.3},
        "Throughput": {"Average": 100.5, "Peak": 150.0},
    }


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
