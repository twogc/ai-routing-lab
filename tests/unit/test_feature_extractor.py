"""Unit tests for FeatureExtractor."""

from datetime import datetime

import numpy as np
import pytest

from models.core.feature_extractor import (
    DomainFeatureExtractor,
    FeatureExtractionResult,
    FeatureExtractor,
)


@pytest.mark.unit
class TestFeatureExtractor:
    """Test suite for FeatureExtractor."""

    def test_initialization(self):
        """Test feature extractor initialization."""
        extractor = FeatureExtractor()

        assert extractor.window_sizes == [5, 10, 20]
        assert extractor.feature_names == []

    def test_initialization_custom_windows(self):
        """Test initialization with custom window sizes."""
        extractor = FeatureExtractor(window_sizes=[3, 7, 15])

        assert extractor.window_sizes == [3, 7, 15]

    def test_extract_time_features(self):
        """Test time feature extraction."""
        extractor = FeatureExtractor()

        # Create timestamps
        timestamps = np.array(
            [
                datetime(2025, 1, 15, 10, 30, 0).timestamp(),
                datetime(2025, 1, 15, 14, 45, 0).timestamp(),
                datetime(2025, 1, 16, 8, 0, 0).timestamp(),
            ]
        )

        features, feature_names = extractor.extract_time_features(timestamps)

        assert features.shape[0] == len(timestamps)
        assert features.shape[1] == 6  # hour, minute, weekday, day, month, is_weekend
        assert len(feature_names) == 6
        assert "hour" in feature_names
        assert "weekday" in feature_names

    def test_extract_statistical_features(self, sample_features):
        """Test statistical feature extraction."""
        extractor = FeatureExtractor()

        features, feature_names = extractor.extract_statistical_features(sample_features)

        assert len(features) > 0
        assert len(feature_names) > 0
        assert any("mean" in name for name in feature_names)
        assert any("std" in name for name in feature_names)

    def test_extract_statistical_features_with_quantiles(self, sample_features):
        """Test statistical features with quantiles."""
        extractor = FeatureExtractor()

        features, feature_names = extractor.extract_statistical_features(
            sample_features, include_quantiles=True
        )

        assert any("q25" in name for name in feature_names)
        assert any("q75" in name for name in feature_names)
        assert any("iqr" in name for name in feature_names)

    def test_extract_rolling_features(self, sample_features):
        """Test rolling feature extraction."""
        extractor = FeatureExtractor(window_sizes=[3, 5])

        features, feature_names = extractor.extract_rolling_features(sample_features)

        # Features may be empty if data is too short
        if len(features) > 0:
            assert len(feature_names) > 0
            assert any("rolling" in name for name in feature_names)

    def test_extract_rolling_features_short_data(self):
        """Test rolling features with short data."""
        extractor = FeatureExtractor(window_sizes=[10])
        X_short = np.random.randn(5, 3)  # Too short for window

        features, feature_names = extractor.extract_rolling_features(X_short)

        # Should handle gracefully
        assert isinstance(features, np.ndarray)

    def test_extract_exponential_moving_average(self, sample_features):
        """Test EMA feature extraction."""
        extractor = FeatureExtractor()

        features, feature_names = extractor.extract_exponential_moving_average(
            sample_features, spans=[3, 5]
        )

        if len(features) > 0:
            assert len(feature_names) > 0
            assert any("ema" in name for name in feature_names)

    def test_extract_all_features(self, sample_features):
        """Test extracting all features."""
        extractor = FeatureExtractor()

        result = extractor.extract_all_features(
            sample_features, include_stats=True, include_rolling=True, include_ema=True
        )

        assert isinstance(result, FeatureExtractionResult)
        assert result.features.shape[0] == sample_features.shape[0]
        assert result.n_features_created > sample_features.shape[1]
        assert len(result.feature_names) == result.n_features_created
        assert result.extraction_time_ms >= 0

    def test_extract_all_features_with_timestamps(self, sample_features):
        """Test extracting all features with timestamps."""
        extractor = FeatureExtractor()

        timestamps = np.array(
            [
                datetime(2025, 1, 15, 10, 30, 0).timestamp(),
                datetime(2025, 1, 15, 14, 45, 0).timestamp(),
                datetime(2025, 1, 16, 8, 0, 0).timestamp(),
                datetime(2025, 1, 16, 12, 0, 0).timestamp(),
                datetime(2025, 1, 17, 9, 0, 0).timestamp(),
            ]
        )

        result = extractor.extract_all_features(
            sample_features, timestamps=timestamps, include_stats=True
        )

        assert result.n_features_created > sample_features.shape[1]
        assert any("hour" in name or "weekday" in name for name in result.feature_names)

    def test_extract_all_features_minimal(self, sample_features):
        """Test extracting minimal features."""
        extractor = FeatureExtractor()

        result = extractor.extract_all_features(
            sample_features, include_stats=False, include_rolling=False, include_ema=False
        )

        # Should at least have original features
        assert result.n_features_created >= sample_features.shape[1]


@pytest.mark.unit
class TestDomainFeatureExtractor:
    """Test suite for DomainFeatureExtractor."""

    def test_initialization(self):
        """Test domain feature extractor initialization."""
        extractor = DomainFeatureExtractor()

        assert isinstance(extractor, FeatureExtractor)

    def test_extract_network_features(self):
        """Test network feature extraction."""
        extractor = DomainFeatureExtractor()

        metrics = {
            "rtt": 50.0,
            "packet_loss": 2.5,
            "bandwidth_used": 500,
            "bandwidth_total": 1000,
            "errors": 10,
            "total_requests": 1000,
        }

        features = extractor.extract_network_features(metrics)

        assert "rtt_zscore" in features
        assert "packet_loss_rate" in features
        assert "bandwidth_utilization" in features
        assert "error_rate" in features
        assert 0 <= features["bandwidth_utilization"] <= 1

    def test_extract_performance_features(self):
        """Test performance feature extraction."""
        extractor = DomainFeatureExtractor()

        metrics = {"throughput": 500, "latency_p95": 25.0, "cpu_usage": 75.0, "memory_usage": 60.0}

        features = extractor.extract_performance_features(metrics)

        assert "throughput_normalized" in features
        assert "latency_p95_normalized" in features
        assert "cpu_usage_percent" in features
        assert "memory_usage_percent" in features
        assert 0 <= features["cpu_usage_percent"] <= 1

    def test_extract_network_features_missing_values(self):
        """Test network features with missing values."""
        extractor = DomainFeatureExtractor()

        metrics = {"rtt": 50.0}  # Missing other values

        features = extractor.extract_network_features(metrics)

        # Should handle missing values gracefully
        assert "rtt_zscore" in features
        assert isinstance(features["packet_loss_rate"], (int, float))
