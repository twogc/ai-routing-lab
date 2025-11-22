"""
Feature Extractor - Generates domain-specific features from raw data.

Creates temporal features, statistical features, and domain-specific indicators
for anomaly detection, load prediction, and route optimization.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FeatureExtractionResult:
    """Result of feature extraction"""

    features: np.ndarray
    feature_names: List[str]
    extraction_time_ms: float
    n_features_created: int


class FeatureExtractor:
    """
    Extracts and engineers features for ML models.

    Feature types:
    - Time features: hour, day, weekday, month
    - Statistical features: mean, std, min, max, median, quantiles
    - Temporal features: rolling means, exponential moving averages
    - Domain features: performance indicators, network metrics
    """

    def __init__(self, window_sizes: List[int] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize Feature Extractor.

        Args:
            window_sizes: Window sizes for rolling statistics
            logger: Optional logger instance
        """
        self.window_sizes = window_sizes or [5, 10, 20]
        self.logger = logger or logging.getLogger(__name__)
        self.feature_names = []

    def extract_time_features(self, timestamps: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Extract temporal features from timestamps.

        Args:
            timestamps: Array of Unix timestamps or datetime objects

        Returns:
            Tuple of (features, feature_names)
        """
        from datetime import datetime

        n_samples = len(timestamps)
        features = []
        feature_names = []

        for ts in timestamps:
            # Convert to datetime if needed
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts)
            else:
                dt = ts

            # Time-of-day features
            hour = dt.hour / 24.0  # Normalize to [0, 1]
            minute = dt.minute / 60.0

            # Day-of-week (0=Monday, 6=Sunday)
            weekday = dt.weekday() / 7.0

            # Day of month
            day = dt.day / 31.0

            # Month
            month = dt.month / 12.0

            # Whether it's weekend
            is_weekend = 1.0 if dt.weekday() >= 5 else 0.0

            features.append([hour, minute, weekday, day, month, is_weekend])

        if not feature_names:
            feature_names = ["hour", "minute", "weekday", "day", "month", "is_weekend"]

        return np.array(features), feature_names

    def extract_statistical_features(
        self, X: np.ndarray, include_quantiles: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract statistical features from data.

        Args:
            X: Input data (n_samples, n_features)
            include_quantiles: Whether to include quantile features

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        feature_names = []

        # For each column, extract statistics
        for col in range(X.shape[1]):
            col_data = X[:, col]

            # Basic statistics
            col_features = [
                np.mean(col_data),
                np.std(col_data),
                np.min(col_data),
                np.max(col_data),
                np.median(col_data),
                np.ptp(col_data),  # Peak-to-peak (max - min)
            ]

            col_feature_names = [
                f"col{col}_mean",
                f"col{col}_std",
                f"col{col}_min",
                f"col{col}_max",
                f"col{col}_median",
                f"col{col}_range",
            ]

            # Add quantiles
            if include_quantiles:
                q25 = np.percentile(col_data, 25)
                q75 = np.percentile(col_data, 75)
                col_features.extend([q25, q75, q75 - q25])
                col_feature_names.extend([f"col{col}_q25", f"col{col}_q75", f"col{col}_iqr"])

            features.extend(col_features)
            feature_names.extend(col_feature_names)

        return np.array(features), feature_names

    def extract_rolling_features(
        self, X: np.ndarray, window_sizes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract rolling window statistics.

        Args:
            X: Input data (n_samples, n_features)
            window_sizes: Sizes of rolling windows

        Returns:
            Tuple of (features, feature_names)
        """
        if window_sizes is None:
            window_sizes = self.window_sizes

        features = []
        feature_names = []

        # For each column
        for col in range(X.shape[1]):
            col_data = X[:, col]

            # For each window size
            for window in window_sizes:
                if len(col_data) < window:
                    continue

                # Rolling mean
                rolling_mean = self._rolling_window(col_data, window, np.mean)
                features.append(rolling_mean)
                feature_names.append(f"col{col}_rolling_mean_w{window}")

                # Rolling std
                rolling_std = self._rolling_window(col_data, window, np.std)
                features.append(rolling_std)
                feature_names.append(f"col{col}_rolling_std_w{window}")

                # Rolling min/max
                rolling_min = self._rolling_window(col_data, window, np.min)
                features.append(rolling_min)
                feature_names.append(f"col{col}_rolling_min_w{window}")

        # Stack features (pad shorter ones)
        if features:
            max_len = max(len(f) for f in features)
            padded_features = []
            for f in features:
                if len(f) < max_len:
                    f = np.concatenate([np.full(max_len - len(f), np.nan), f])
                padded_features.append(f)
            features = np.column_stack(padded_features)
        else:
            features = np.array([])

        return features, feature_names

    def extract_exponential_moving_average(
        self, X: np.ndarray, spans: List[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract exponential moving average features.

        Args:
            X: Input data (n_samples, n_features)
            spans: Spans for EMA calculation

        Returns:
            Tuple of (features, feature_names)
        """
        if spans is None:
            spans = [5, 10, 20]

        features = []
        feature_names = []

        for col in range(X.shape[1]):
            col_data = X[:, col]

            for span in spans:
                # Calculate EMA
                ema = self._exponential_moving_average(col_data, span)
                features.append(ema)
                feature_names.append(f"col{col}_ema_span{span}")

        if features:
            features = np.column_stack(features)
        else:
            features = np.array([])

        return features, feature_names

    def extract_all_features(
        self,
        X: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        include_stats: bool = True,
        include_rolling: bool = True,
        include_ema: bool = True,
    ) -> FeatureExtractionResult:
        """
        Extract all available features.

        Args:
            X: Input data (n_samples, n_features)
            timestamps: Optional timestamps for time features
            include_stats: Whether to include statistical features
            include_rolling: Whether to include rolling features
            include_ema: Whether to include EMA features

        Returns:
            FeatureExtractionResult
        """
        import time as time_module

        start_time = time_module.time()
        all_features = []
        all_feature_names = []

        # Original features
        all_features.append(X)
        for i in range(X.shape[1]):
            all_feature_names.append(f"original_feat_{i}")

        # Time features
        if timestamps is not None:
            time_feats, time_names = self.extract_time_features(timestamps)
            if len(time_feats) == len(X):
                all_features.append(time_feats)
                all_feature_names.extend(time_names)

        # Statistical features (computed per-sample)
        if include_stats:
            stat_feats, stat_names = self.extract_statistical_features(X)
            if len(stat_feats) > 0:
                # Reshape to match n_samples
                if stat_feats.ndim == 1:
                    stat_feats = np.tile(stat_feats, (X.shape[0], 1))
                all_features.append(stat_feats)
                all_feature_names.extend(stat_names)

        # Rolling features
        if include_rolling:
            rolling_feats, rolling_names = self.extract_rolling_features(X)
            if len(rolling_feats) > 0:
                all_features.append(rolling_feats)
                all_feature_names.extend(rolling_names)

        # EMA features
        if include_ema:
            ema_feats, ema_names = self.extract_exponential_moving_average(X)
            if len(ema_feats) > 0:
                all_features.append(ema_feats)
                all_feature_names.extend(ema_names)

        # Combine all features
        combined_features = np.column_stack(all_features)

        extraction_time_ms = (time_module.time() - start_time) * 1000

        result = FeatureExtractionResult(
            features=combined_features,
            feature_names=all_feature_names,
            extraction_time_ms=extraction_time_ms,
            n_features_created=combined_features.shape[1],
        )

        self.logger.info(
            f"Extracted {result.n_features_created} features " f"in {extraction_time_ms:.2f}ms"
        )

        return result

    @staticmethod
    def _rolling_window(data: np.ndarray, window: int, func) -> np.ndarray:
        """Apply rolling window function"""
        if len(data) < window:
            return np.array([])

        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = func(data[i - window + 1 : i + 1])

        return result

    @staticmethod
    def _exponential_moving_average(data: np.ndarray, span: int) -> np.ndarray:
        """Calculate exponential moving average"""
        alpha = 2 / (span + 1)
        ema = np.full(len(data), np.nan)
        ema[0] = data[0]

        for i in range(1, len(data)):
            if not np.isnan(data[i]):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema


class DomainFeatureExtractor(FeatureExtractor):
    """Extracts domain-specific features for CloudBridge"""

    def extract_network_features(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Extract network-specific features"""
        return {
            "rtt_zscore": (metrics.get("rtt", 0) - 50) / 10,  # Normalize RTT
            "packet_loss_rate": metrics.get("packet_loss", 0) / 100,
            "bandwidth_utilization": metrics.get("bandwidth_used", 0)
            / metrics.get("bandwidth_total", 1),
            "error_rate": metrics.get("errors", 0) / metrics.get("total_requests", 1),
        }

    def extract_performance_features(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Extract performance indicators"""
        return {
            "throughput_normalized": metrics.get("throughput", 0) / 1000,  # Normalize to Gbps
            "latency_p95_normalized": metrics.get("latency_p95", 0) / 100,  # Normalize to 100ms
            "cpu_usage_percent": metrics.get("cpu_usage", 0) / 100,
            "memory_usage_percent": metrics.get("memory_usage", 0) / 100,
        }
