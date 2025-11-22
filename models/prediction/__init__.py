"""Prediction models for AI Routing Lab."""

# Latency and Jitter prediction models
from .arima_model import ARIMAModel
from .jitter_predictor import JitterPrediction, JitterPredictor
from .latency_predictor import LatencyPrediction, LatencyPredictor

# Legacy models from CloudBridge AI Service (may need adaptation)
from .load_ensemble import EnsembleForecast, LoadPredictionEnsemble
from .lstm_forecast import LSTMForecastModel
from .prophet_model import ProphetModel
from .random_forest_load import RandomForestLoadModel
from .route_prediction_ensemble import RoutePrediction, RoutePredictionEnsemble

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
