"""Prediction models for AI Routing Lab."""

# Latency and Jitter prediction models
from .latency_predictor import LatencyPredictor, LatencyPrediction
from .jitter_predictor import JitterPredictor, JitterPrediction
from .route_prediction_ensemble import RoutePredictionEnsemble, RoutePrediction

# Legacy models from CloudBridge AI Service (may need adaptation)
from .load_ensemble import LoadPredictionEnsemble, EnsembleForecast
from .random_forest_load import RandomForestLoadModel
from .lstm_forecast import LSTMForecastModel
from .arima_model import ARIMAModel
from .prophet_model import ProphetModel

__all__ = [
    # New models for latency/jitter prediction
    "LatencyPredictor",
    "LatencyPrediction",
    "JitterPredictor",
    "JitterPrediction",
    "RoutePredictionEnsemble",
    "RoutePrediction",
    # Legacy models
    "LoadPredictionEnsemble",
    "EnsembleForecast",
    "RandomForestLoadModel",
    "LSTMForecastModel",
    "ARIMAModel",
    "ProphetModel",
]
