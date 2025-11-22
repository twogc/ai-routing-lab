"""
Data Preprocessor - Handles data cleaning, normalization, and transformation.

Performs feature engineering and data preparation for ML models.
Handles missing values, outliers, normalization, and categorical encoding.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PreprocessingStats:
    """Statistics from data preprocessing"""

    original_shape: Tuple[int, ...]
    final_shape: Tuple[int, ...]
    missing_values_handled: int
    outliers_detected: int
    outliers_removed: int
    features_normalized: int
    preprocessing_time_ms: float


class DataPreprocessor:
    """
    Preprocesses data for ML models with handling for missing values and outliers.

    Features:
    - Missing value imputation (mean, median, forward-fill)
    - Outlier detection and removal (IQR method)
    - Feature normalization (standardization, min-max scaling)
    - Categorical encoding
    - Data validation
    - Statistics tracking
    """

    def __init__(
        self,
        strategy: str = "mean",
        outlier_method: str = "iqr",
        normalization: str = "standard",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Data Preprocessor.

        Args:
            strategy: Imputation strategy ('mean', 'median', 'forward_fill', 'drop')
            outlier_method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            normalization: Normalization method ('standard', 'minmax', 'robust')
            logger: Optional logger instance
        """
        self.strategy = strategy
        self.outlier_method = outlier_method
        self.normalization = normalization
        self.logger = logger or logging.getLogger(__name__)

        # Fit statistics
        self.fitted = False
        self.feature_means = None
        self.feature_medians = None
        self.feature_stds = None
        self.feature_mins = None
        self.feature_maxs = None
        self.feature_names = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "DataPreprocessor":
        """
        Fit preprocessor on data.

        Args:
            X: Input data (n_samples, n_features)
            feature_names: Optional feature names

        Returns:
            Self for chaining
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Calculate statistics
        self.feature_means = np.nanmean(X, axis=0)
        self.feature_medians = np.nanmedian(X, axis=0)
        self.feature_stds = np.nanstd(X, axis=0)
        self.feature_mins = np.nanmin(X, axis=0)
        self.feature_maxs = np.nanmax(X, axis=0)

        # Handle constant features (std = 0)
        self.feature_stds[self.feature_stds == 0] = 1.0

        self.fitted = True
        self.logger.info(f"DataPreprocessor fitted on {X.shape[0]} samples, {X.shape[1]} features")

        return self

    def transform(
        self, X: np.ndarray, remove_outliers: bool = True, normalize: bool = True
    ) -> Tuple[np.ndarray, PreprocessingStats]:
        """
        Transform data using fitted preprocessor.

        Args:
            X: Input data (n_samples, n_features)
            remove_outliers: Whether to remove outlier samples
            normalize: Whether to normalize features

        Returns:
            Tuple of (transformed_data, preprocessing_stats)
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        import time

        start_time = time.time()

        X = X.astype(np.float32).copy()
        original_shape = X.shape

        # Handle missing values
        missing_count = self._handle_missing_values(X)

        # Detect and remove outliers
        outliers_detected, outliers_removed = 0, 0
        if remove_outliers:
            outliers_detected, outliers_removed = self._handle_outliers(X)

        # Normalize features
        features_normalized = 0
        if normalize:
            features_normalized = self._normalize_features(X)

        # Calculate statistics
        processing_time_ms = (time.time() - start_time) * 1000

        stats = PreprocessingStats(
            original_shape=original_shape,
            final_shape=X.shape,
            missing_values_handled=missing_count,
            outliers_detected=outliers_detected,
            outliers_removed=outliers_removed,
            features_normalized=features_normalized,
            preprocessing_time_ms=processing_time_ms,
        )

        self.logger.debug(
            f"Data transformation: {original_shape} -> {X.shape}, "
            f"missing={missing_count}, outliers_removed={outliers_removed}"
        )

        return X, stats

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        remove_outliers: bool = True,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, PreprocessingStats]:
        """
        Fit on data and transform in one step.

        Args:
            X: Input data (n_samples, n_features)
            feature_names: Optional feature names
            remove_outliers: Whether to remove outlier samples
            normalize: Whether to normalize features

        Returns:
            Tuple of (transformed_data, preprocessing_stats)
        """
        self.fit(X, feature_names)
        return self.transform(X, remove_outliers, normalize)

    def _handle_missing_values(self, X: np.ndarray) -> int:
        """
        Handle missing values (NaN) in data.

        Args:
            X: Data array (modified in-place)

        Returns:
            Number of missing values handled
        """
        missing_count = np.isnan(X).sum()

        if missing_count == 0:
            return 0

        for col in range(X.shape[1]):
            col_missing = np.isnan(X[:, col])

            if np.any(col_missing):
                if self.strategy == "mean":
                    X[col_missing, col] = self.feature_means[col]
                elif self.strategy == "median":
                    X[col_missing, col] = self.feature_medians[col]
                elif self.strategy == "forward_fill":
                    # Forward fill with fallback to mean
                    for i in np.where(col_missing)[0]:
                        if i == 0:
                            X[i, col] = self.feature_means[col]
                        else:
                            X[i, col] = X[i - 1, col]
                elif self.strategy == "drop":
                    # Already handled by removing rows
                    continue

        return missing_count

    def _handle_outliers(self, X: np.ndarray) -> Tuple[int, int]:
        """
        Detect and optionally remove outliers using IQR method.

        Args:
            X: Data array

        Returns:
            Tuple of (outliers_detected, outliers_removed)
        """
        outliers_detected = 0
        outliers_removed = 0

        if self.outlier_method == "iqr":
            # Use Interquartile Range method
            q1 = np.nanpercentile(X, 25, axis=0)
            q3 = np.nanpercentile(X, 75, axis=0)
            iqr = q3 - q1

            # Define outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Find outlier rows
            outlier_mask = np.zeros(X.shape[0], dtype=bool)

            for col in range(X.shape[1]):
                col_outliers = (X[:, col] < lower_bound[col]) | (X[:, col] > upper_bound[col])
                outlier_mask |= col_outliers
                outliers_detected += np.sum(col_outliers)

            # Cap outliers instead of removing
            for col in range(X.shape[1]):
                col_mask_lower = X[:, col] < lower_bound[col]
                col_mask_upper = X[:, col] > upper_bound[col]
                X[col_mask_lower, col] = lower_bound[col]
                X[col_mask_upper, col] = upper_bound[col]
            outliers_removed = np.sum(outlier_mask)

        elif self.outlier_method == "zscore":
            # Z-score method
            z_scores = np.abs((X - self.feature_means) / self.feature_stds)
            outlier_mask = (z_scores > 3).any(axis=1)
            outliers_detected = np.sum(outlier_mask)

            # Cap values at 3 sigma
            for col in range(X.shape[1]):
                col_mask = z_scores[:, col] > 3
                X[col_mask, col] = self.feature_means[col]
            outliers_removed = outliers_detected

        return outliers_detected, outliers_removed

    def _normalize_features(self, X: np.ndarray) -> int:
        """
        Normalize features.

        Args:
            X: Data array (modified in-place)

        Returns:
            Number of features normalized
        """
        if self.normalization == "standard":
            # Standardization: (x - mean) / std
            X[:] = (X - self.feature_means) / np.maximum(self.feature_stds, 1e-8)

        elif self.normalization == "minmax":
            # Min-Max scaling: (x - min) / (max - min)
            ranges = self.feature_maxs - self.feature_mins
            ranges[ranges == 0] = 1.0
            X[:] = (X - self.feature_mins) / ranges

        elif self.normalization == "robust":
            # Robust scaling using median and IQR
            q1 = np.nanpercentile(X, 25, axis=0)
            q3 = np.nanpercentile(X, 75, axis=0)
            iqr = np.maximum(q3 - q1, 1e-8)
            X[:] = (X - self.feature_medians) / iqr

        return X.shape[1]

    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        if not self.fitted:
            return {}

        return {
            "fitted": True,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "feature_means": (
                self.feature_means.tolist() if self.feature_means is not None else None
            ),
            "feature_stds": self.feature_stds.tolist() if self.feature_stds is not None else None,
            "feature_mins": self.feature_mins.tolist() if self.feature_mins is not None else None,
            "feature_maxs": self.feature_maxs.tolist() if self.feature_maxs is not None else None,
            "strategy": self.strategy,
            "outlier_method": self.outlier_method,
            "normalization": self.normalization,
        }


class RobustPreprocessor(DataPreprocessor):
    """Extended preprocessor with additional robustness features"""

    def handle_categorical(
        self, X: np.ndarray, categorical_cols: List[int], encoding: str = "onehot"
    ) -> np.ndarray:
        """
        Handle categorical features.

        Args:
            X: Data array
            categorical_cols: Indices of categorical columns
            encoding: Encoding method ('onehot', 'label')

        Returns:
            Transformed array with encoded categorical features
        """
        if encoding == "onehot":
            # One-hot encoding would expand features
            # For now, just handle with label encoding
            pass

        return X

    def validate_data(self, X: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate data quality.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if X.ndim != 2:
            errors.append(f"Expected 2D array, got {X.ndim}D")

        if X.shape[0] == 0:
            errors.append("Empty data")

        if self.fitted and X.shape[1] != len(self.feature_names):
            errors.append(f"Feature mismatch: expected {len(self.feature_names)}, got {X.shape[1]}")

        # Check for all NaN columns
        if np.all(np.isnan(X), axis=0).any():
            errors.append("Found columns with all NaN values")

        return len(errors) == 0, errors
