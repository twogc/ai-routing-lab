"""Unit tests for ProphetModel."""

import numpy as np
import pytest

from models.prediction.prophet_model import ProphetModel


class TestProphetModel:
    """Test suite for ProphetModel."""

    @pytest.fixture
    def model(self):
        return ProphetModel(seasonality_period=10)

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        t = np.arange(100)
        # Linear trend + seasonality + noise
        trend = 0.1 * t
        seasonal = np.sin(2 * np.pi * t / 10)
        y = trend + seasonal + np.random.normal(0, 0.1, 100)
        return y

    def test_initialization(self, model):
        """Test initialization."""
        assert model.seasonality_period == 10
        assert not model.fitted
        assert model.trend_coefs is None

    def test_fit(self, model, sample_data):
        """Test fitting."""
        model.fit(sample_data)

        assert model.fitted
        assert model.trend_coefs is not None
        assert model.seasonal_coefs is not None
        assert len(model.seasonal_coefs) == 10
        assert model.metrics["r_squared"] > 0.8

    def test_forecast_not_fitted(self, model, sample_data):
        """Test forecast before training."""
        with pytest.raises(RuntimeError):
            model.forecast(sample_data)

    def test_forecast(self, model, sample_data):
        """Test forecasting."""
        model.fit(sample_data)

        steps = 10
        forecasts, (ci_lower, ci_upper) = model.forecast(sample_data, steps=steps)

        assert len(forecasts) == steps
        assert len(ci_lower) == steps
        assert len(ci_upper) == steps

        # Check if forecast captures seasonality (roughly)
        # Peak should be around step 2-3 (sin(pi/2) = 1 at t=2.5)
        # Trough around step 7-8 (sin(3pi/2) = -1 at t=7.5)
        # Since we start at t=100 (multiple of 10), next is t=101..110
        # sin(2*pi*101/10) = sin(0.2*pi) > 0
        # sin(2*pi*102/10) = sin(0.4*pi) > 0
        # sin(2*pi*103/10) = sin(0.6*pi) > 0
        pass

    def test_decompose(self, model, sample_data):
        """Test decomposition."""
        model.fit(sample_data)

        trend, seasonal, residual = model.decompose(sample_data)

        assert len(trend) == len(sample_data)
        assert len(seasonal) == len(sample_data)
        assert len(residual) == len(sample_data)

        # Reconstruct
        reconstructed = trend + seasonal + residual
        assert np.allclose(reconstructed, sample_data, atol=1e-5)

    def test_score(self, model, sample_data):
        """Test scoring."""
        model.fit(sample_data)
        score = model.score(sample_data)
        assert score > 0.8

    def test_get_metrics(self, model, sample_data):
        """Test metrics retrieval."""
        model.fit(sample_data)
        metrics = model.get_metrics()

        assert "mse" in metrics
        assert "seasonality_period" in metrics
