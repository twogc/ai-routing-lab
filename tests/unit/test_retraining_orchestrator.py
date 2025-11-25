"""Unit tests for RetrainingOrchestrator."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models.monitoring.retraining_orchestrator import RetrainingJob, RetrainingOrchestrator


class TestRetrainingOrchestrator:
    """Test suite for RetrainingOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        return RetrainingOrchestrator(retraining_interval_hours=24)

    def test_initialization(self, orchestrator):
        """Test initialization."""
        assert orchestrator.retraining_interval_hours == 24
        assert len(orchestrator.jobs) == 0
        assert len(orchestrator.active_jobs) == 0

    def test_schedule_retraining_immediate(self, orchestrator):
        """Test immediate scheduling."""
        job_id = orchestrator.schedule_retraining("model_1", immediate=True)

        assert job_id is not None
        assert "model_1" in orchestrator.active_jobs
        assert orchestrator.active_jobs["model_1"] == job_id
        assert len(orchestrator.jobs) == 1

    def test_schedule_retraining_interval(self, orchestrator):
        """Test interval-based scheduling."""
        # First run
        job_id = orchestrator.schedule_retraining("model_1")
        assert job_id is not None

        # Simulate completion
        orchestrator.last_retraining["model_1"] = datetime.now()
        del orchestrator.active_jobs["model_1"]

        # Second run (too soon)
        job_id_2 = orchestrator.schedule_retraining("model_1")
        assert job_id_2 is None

        # Second run (after interval)
        with patch("models.monitoring.retraining_orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime.now() + timedelta(hours=25)
            job_id_3 = orchestrator.schedule_retraining("model_1")
            assert job_id_3 is not None

    def test_execute_retraining(self, orchestrator):
        """Test retraining execution."""
        job_id = orchestrator.schedule_retraining("model_1", immediate=True)
        X = np.zeros((10, 5))
        y = np.zeros(10)

        result = orchestrator.execute_retraining(job_id, X, y)

        assert result["status"] == "completed"
        assert result["training_samples"] == 10
        assert "model_1" not in orchestrator.active_jobs
        assert "model_1" in orchestrator.last_retraining

    def test_execute_retraining_invalid_job(self, orchestrator):
        """Test execution with invalid job ID."""
        with pytest.raises(ValueError):
            orchestrator.execute_retraining("invalid_id", np.zeros(1), np.zeros(1))

    def test_should_retrain_on_drift(self, orchestrator):
        """Test drift trigger."""
        assert orchestrator.should_retrain_on_drift("model_1", 0.1)  # > 0.05
        assert not orchestrator.should_retrain_on_drift("model_1", 0.01)  # < 0.05

    def test_should_retrain_on_performance(self, orchestrator):
        """Test performance trigger."""
        # Degradation 0.1 > 0.05
        assert orchestrator.should_retrain_on_performance("model_1", 0.8, 0.9)
        # Degradation 0.01 < 0.05
        assert not orchestrator.should_retrain_on_performance("model_1", 0.89, 0.9)

    def test_get_job_status(self, orchestrator):
        """Test job status retrieval."""
        job_id = orchestrator.schedule_retraining("model_1", immediate=True)
        status = orchestrator.get_job_status(job_id)

        assert status is not None
        assert status["job_id"] == job_id
        assert status["status"] == "pending"

    def test_get_retraining_history(self, orchestrator):
        """Test history retrieval."""
        orchestrator.schedule_retraining("model_1", immediate=True)
        orchestrator.schedule_retraining("model_2", immediate=True)

        history = orchestrator.get_retraining_history()
        assert len(history) == 2

        history_m1 = orchestrator.get_retraining_history(model_id="model_1")
        assert len(history_m1) == 1

    def test_get_metrics(self, orchestrator):
        """Test metrics retrieval."""
        job_id = orchestrator.schedule_retraining("model_1", immediate=True)
        orchestrator.execute_retraining(job_id, np.zeros((10, 5)), np.zeros(10))

        metrics = orchestrator.get_metrics()
        assert metrics["total_jobs"] == 1
        assert metrics["completed_jobs"] == 1
        assert metrics["active_jobs"] == 0
