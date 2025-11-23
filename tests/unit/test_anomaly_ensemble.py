"""Unit tests for AnomalyEnsemble."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from models.anomaly.anomaly_ensemble import AnomalyEnsemble, EnsembleAnomalyPrediction, AnomalyDetectionPipeline

class TestAnomalyEnsemble:
    """Test suite for AnomalyEnsemble."""

    @pytest.fixture
    def mock_if(self):
        mock = MagicMock()
        mock.predict.return_value = (np.array([0]), np.array([0.2]))
        mock.predict_sample.return_value = MagicMock(is_anomaly=False, confidence=0.8)
        mock.get_metrics.return_value = {}
        return mock

    @pytest.fixture
    def mock_svm(self):
        mock = MagicMock()
        mock.predict.return_value = (np.array([0]), np.array([0.3]))
        mock.predict_sample.return_value = MagicMock(is_anomaly=False, confidence=0.7)
        mock.get_metrics.return_value = {}
        return mock

    @pytest.fixture
    def mock_lstm(self):
        mock = MagicMock()
        mock.predict.return_value = (np.array([1]), np.array([0.8]))
        mock.predict_sample.return_value = MagicMock(is_anomaly=True, confidence=0.9)
        mock.get_metrics.return_value = {}
        return mock

    @pytest.fixture
    def ensemble(self, mock_if, mock_svm, mock_lstm):
        with patch("models.anomaly.anomaly_ensemble.IsolationForestModel", return_value=mock_if), \
             patch("models.anomaly.anomaly_ensemble.OneClassSVMModel", return_value=mock_svm), \
             patch("models.anomaly.anomaly_ensemble.LSTMAutoencoderModel", return_value=mock_lstm):
            
            ensemble = AnomalyEnsemble()
            ensemble.if_model = mock_if
            ensemble.svm_model = mock_svm
            ensemble.lstm_model = mock_lstm
            return ensemble

    def test_initialization(self, ensemble):
        """Test initialization."""
        assert not ensemble.fitted
        # Check weights sum to 1
        total = ensemble.if_weight + ensemble.svm_weight + ensemble.lstm_weight
        assert abs(total - 1.0) < 1e-6

    def test_fit(self, ensemble):
        """Test training."""
        X = np.zeros((5, 4))
        ensemble.fit(X)
        
        assert ensemble.fitted
        ensemble.if_model.fit.assert_called_once()
        ensemble.svm_model.fit.assert_called_once()
        ensemble.lstm_model.fit.assert_called_once()

    def test_predict_not_fitted(self, ensemble):
        """Test prediction before training."""
        with pytest.raises(RuntimeError):
            ensemble.predict(np.zeros((1, 4)))

    def test_predict(self, ensemble):
        """Test prediction logic."""
        ensemble.fitted = True
        X = np.zeros((1, 4))
        
        # IF(0.25)*0.2 + SVM(0.25)*0.3 + LSTM(0.5)*0.8
        # Scores are normalized inside predict, so we need to mock normalized scores or check logic
        # The mock returns fixed scores.
        # normalize([0.2]) -> [0.5] (if max==min)
        
        preds, scores = ensemble.predict(X)
        
        assert len(preds) == 1
        assert len(scores) == 1

    def test_predict_sample(self, ensemble):
        """Test single sample prediction."""
        ensemble.fitted = True
        X = np.zeros(4)
        
        prediction = ensemble.predict_sample(X)
        
        assert isinstance(prediction, EnsembleAnomalyPrediction)
        assert len(prediction.voting_results) == 3

    def test_score(self, ensemble):
        """Test scoring."""
        ensemble.fitted = True
        X = np.zeros((1, 4))
        y = np.array([0])
        
        score = ensemble.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_get_metrics(self, ensemble):
        """Test metrics retrieval."""
        ensemble.fitted = True
        ensemble.predict(np.zeros((1, 4)))
        
        metrics = ensemble.get_metrics()
        assert "anomaly_rate_percent" in metrics
        assert "weights" in metrics

    def test_normalize_scores(self, ensemble):
        """Test score normalization."""
        scores = np.array([0.0, 0.5, 1.0])
        norm = ensemble._normalize_scores(scores)
        assert np.min(norm) == 0.0
        assert np.max(norm) == 1.0
        
        # Test constant scores
        scores = np.array([0.5, 0.5])
        norm = ensemble._normalize_scores(scores)
        assert np.all(norm == 0.5)

class TestAnomalyDetectionPipeline:
    """Test suite for AnomalyDetectionPipeline."""

    def test_pipeline(self):
        """Test pipeline flow."""
        mock_ensemble = MagicMock()
        mock_ensemble.predict_sample.return_value = MagicMock()
        
        pipeline = AnomalyDetectionPipeline(ensemble=mock_ensemble)
        X = np.zeros((5, 4))
        
        pipeline.fit(X)
        mock_ensemble.fit.assert_called_once()
        
        pipeline.predict(X[0])
        mock_ensemble.predict_sample.assert_called_once()
