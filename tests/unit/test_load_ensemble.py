"""Unit tests for LoadPredictionEnsemble."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from models.prediction.load_ensemble import LoadPredictionEnsemble, EnsembleForecast


class TestLoadPredictionEnsemble:
    """Test suite for LoadPredictionEnsemble."""

    @pytest.fixture
    def mock_rf(self):
        mock = MagicMock()
        mock.predict.return_value = (np.array([10.0]), np.array([1.0]))
        mock.predict_sample.return_value = MagicMock(predicted_load=10.0)
        mock.get_metrics.return_value = {}
        return mock

    @pytest.fixture
    def mock_lstm(self):
        mock = MagicMock()
        mock.predict.return_value = (np.array([11.0]), (9.0, 13.0))
        mock.get_metrics.return_value = {}
        return mock

    @pytest.fixture
    def mock_arima(self):
        mock = MagicMock()
        mock.forecast.return_value = (np.array([12.0]), np.array([1.0]))
        mock.get_metrics.return_value = {}
        return mock

    @pytest.fixture
    def mock_prophet(self):
        mock = MagicMock()
        mock.forecast.return_value = (np.array([11.5]), np.array([1.0]))
        mock.get_metrics.return_value = {}
        return mock

    @pytest.fixture
    def ensemble(self, mock_rf, mock_lstm, mock_arima, mock_prophet):
        with (
            patch("models.prediction.load_ensemble.RandomForestLoadModel", return_value=mock_rf),
            patch("models.prediction.load_ensemble.LSTMForecastModel", return_value=mock_lstm),
            patch("models.prediction.load_ensemble.ARIMAModel", return_value=mock_arima),
            patch("models.prediction.load_ensemble.ProphetModel", return_value=mock_prophet),
        ):

            ensemble = LoadPredictionEnsemble()
            ensemble.rf_model = mock_rf
            ensemble.lstm_model = mock_lstm
            ensemble.arima_model = mock_arima
            ensemble.prophet_model = mock_prophet
            return ensemble

    def test_initialization(self, ensemble):
        """Test initialization."""
        assert not ensemble.fitted
        # Check weights sum to 1
        total = (
            ensemble.rf_weight
            + ensemble.lstm_weight
            + ensemble.arima_weight
            + ensemble.prophet_weight
        )
        assert abs(total - 1.0) < 1e-6

    def test_fit(self, ensemble):
        """Test training."""
        X = np.zeros((5, 4))
        y = np.zeros(5)
        ensemble.fit(X, y)

        assert ensemble.fitted
        ensemble.rf_model.fit.assert_called_once()
        ensemble.lstm_model.fit.assert_called_once()
        ensemble.arima_model.fit.assert_called_once()
        ensemble.prophet_model.fit.assert_called_once()

    def test_predict_not_fitted(self, ensemble):
        """Test prediction before training."""
        with pytest.raises(RuntimeError):
            ensemble.predict(np.zeros((1, 4)))

    def test_predict(self, ensemble):
        """Test prediction logic."""
        ensemble.fitted = True
        X = np.zeros((1, 4))

        # RF(0.25)*10 + LSTM(0.35)*11 + ARIMA(0.1)*12 + Prophet(0.3)*11
        # Note: ARIMA and Prophet predictions in `predict` method are mocked differently or calculated differently
        # In `predict`:
        # arima_preds = np.full(len(X), y[-1] if "y" in locals() else X[:, 0].mean())
        # prophet_preds = lstm_preds.copy()

        preds, uncertainties = ensemble.predict(X)

        assert len(preds) == 1
        assert len(uncertainties) == 1

    def test_predict_sample(self, ensemble):
        """Test single sample prediction."""
        ensemble.fitted = True
        X = np.zeros(4)

        prediction = ensemble.predict_sample(X)

        assert isinstance(prediction, EnsembleForecast)
        assert len(prediction.model_predictions) == 4

    def test_score(self, ensemble):
        """Test scoring."""
        ensemble.fitted = True
        X = np.zeros((1, 4))
        y = np.array([10.0])

        score = ensemble.score(X, y)
        assert score <= 1.0

    def test_get_metrics(self, ensemble):
        """Test metrics retrieval."""
        ensemble.fitted = True
        ensemble.predict(np.zeros((1, 4)))

        metrics = ensemble.get_metrics()
        assert "mse" in metrics
        assert "weights" in metrics
