"""Unit tests for DataPreprocessor."""

import numpy as np
import pytest

from models.core.data_preprocessor import DataPreprocessor, PreprocessingStats, RobustPreprocessor


@pytest.mark.unit
class TestDataPreprocessor:
    """Test suite for DataPreprocessor."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor()

        assert preprocessor.strategy == "mean"
        assert preprocessor.outlier_method == "iqr"
        assert preprocessor.normalization == "standard"
        assert not preprocessor.fitted

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        preprocessor = DataPreprocessor(
            strategy="median", outlier_method="zscore", normalization="minmax"
        )

        assert preprocessor.strategy == "median"
        assert preprocessor.outlier_method == "zscore"
        assert preprocessor.normalization == "minmax"

    def test_fit(self, sample_features):
        """Test fitting preprocessor."""
        preprocessor = DataPreprocessor()
        result = preprocessor.fit(sample_features)

        assert result is preprocessor
        assert preprocessor.fitted
        assert preprocessor.feature_means is not None
        assert preprocessor.feature_stds is not None

    def test_fit_with_feature_names(self, sample_features):
        """Test fitting with feature names."""
        feature_names = ["feat1", "feat2", "feat3", "feat4"]
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_features, feature_names=feature_names)

        assert preprocessor.feature_names == feature_names

    def test_fit_raises_error_on_1d(self):
        """Test that fit raises error on 1D array."""
        preprocessor = DataPreprocessor()
        X_1d = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="Expected 2D array"):
            preprocessor.fit(X_1d)

    def test_transform_without_fit_raises_error(self, sample_features):
        """Test that transform raises error without fit."""
        preprocessor = DataPreprocessor()

        with pytest.raises(RuntimeError, match="must be fitted"):
            preprocessor.transform(sample_features)

    def test_fit_transform(self, sample_features):
        """Test fit_transform method."""
        preprocessor = DataPreprocessor()
        X_transformed, stats = preprocessor.fit_transform(sample_features)

        assert X_transformed.shape == sample_features.shape
        assert isinstance(stats, PreprocessingStats)
        assert stats.original_shape == sample_features.shape

    def test_transform_handles_missing_values(self):
        """Test handling of missing values."""
        X = np.array([[1.0, 2.0, 3.0], [np.nan, 4.0, 5.0], [6.0, np.nan, 7.0], [8.0, 9.0, 10.0]])

        preprocessor = DataPreprocessor(strategy="mean")
        X_transformed, stats = preprocessor.fit_transform(X)

        assert not np.isnan(X_transformed).any()
        assert stats.missing_values_handled > 0

    def test_transform_handles_outliers(self):
        """Test outlier handling."""
        X = np.array(
            [[1.0, 2.0, 3.0], [100.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]  # Outlier
        )

        preprocessor = DataPreprocessor(outlier_method="iqr")
        X_transformed, stats = preprocessor.fit_transform(X, remove_outliers=True)

        assert stats.outliers_detected > 0

    def test_normalization_standard(self, sample_features):
        """Test standard normalization."""
        preprocessor = DataPreprocessor(normalization="standard")
        X_transformed, _ = preprocessor.fit_transform(sample_features, normalize=True)

        # Check that features are normalized (mean ~0, std ~1)
        assert np.allclose(X_transformed.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(X_transformed.std(axis=0), 1, atol=1e-6)

    def test_normalization_minmax(self, sample_features):
        """Test min-max normalization."""
        preprocessor = DataPreprocessor(normalization="minmax")
        X_transformed, _ = preprocessor.fit_transform(sample_features, normalize=True)

        # Check that values are in [0, 1] range (with small tolerance for floating point errors)
        assert X_transformed.min() >= -1e-6, f"Min value {X_transformed.min()} is too negative"
        assert X_transformed.max() <= 1 + 1e-6, f"Max value {X_transformed.max()} is too large"

    def test_get_stats(self, sample_features):
        """Test getting preprocessing statistics."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_features)

        stats = preprocessor.get_stats()

        assert stats["fitted"] is True
        assert "n_features" in stats
        assert "feature_names" in stats
        assert "feature_means" in stats

    def test_get_stats_before_fit(self):
        """Test get_stats before fitting."""
        preprocessor = DataPreprocessor()
        stats = preprocessor.get_stats()

        assert stats == {}


@pytest.mark.unit
class TestRobustPreprocessor:
    """Test suite for RobustPreprocessor."""

    def test_initialization(self):
        """Test robust preprocessor initialization."""
        preprocessor = RobustPreprocessor()

        assert isinstance(preprocessor, DataPreprocessor)

    def test_validate_data(self, sample_features):
        """Test data validation."""
        preprocessor = RobustPreprocessor()
        preprocessor.fit(sample_features)

        is_valid, errors = preprocessor.validate_data(sample_features)

        assert is_valid
        assert len(errors) == 0

    def test_validate_data_invalid_shape(self):
        """Test validation with invalid shape."""
        preprocessor = RobustPreprocessor()
        preprocessor.fit(np.random.randn(10, 4))

        X_invalid = np.random.randn(10, 5)  # Wrong number of features
        is_valid, errors = preprocessor.validate_data(X_invalid)

        assert not is_valid
        assert len(errors) > 0

    def test_validate_data_empty(self):
        """Test validation with empty data."""
        preprocessor = RobustPreprocessor()

        X_empty = np.array([]).reshape(0, 4)
        is_valid, errors = preprocessor.validate_data(X_empty)

        assert not is_valid
        assert any("Empty" in err for err in errors)

    def test_handle_categorical(self, sample_features):
        """Test categorical feature handling."""
        preprocessor = RobustPreprocessor()
        preprocessor.fit(sample_features)

        # This is a placeholder - actual implementation may vary
        result = preprocessor.handle_categorical(sample_features, categorical_cols=[0])

        assert result.shape[0] == sample_features.shape[0]
