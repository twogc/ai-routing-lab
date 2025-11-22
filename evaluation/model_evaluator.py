"""
Model Evaluator - Comprehensive evaluation of ML models.

Provides metrics, cross-validation, and performance analysis.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a model."""

    r2_score: float
    mae: float
    rmse: float
    mape: float
    mean_error: float
    std_error: float
    max_error: float
    min_error: float


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    metrics: EvaluationMetrics
    predictions: np.ndarray
    actual: np.ndarray
    errors: np.ndarray
    feature_importance: Optional[Dict[str, float]] = None
    cross_validation_scores: Optional[List[float]] = None
    evaluation_time_ms: float = 0.0


class ModelEvaluator:
    """
    Evaluates ML models with comprehensive metrics.

    Features:
    - Multiple evaluation metrics (R², MAE, RMSE, MAPE)
    - Cross-validation
    - Error analysis
    - Feature importance analysis
    """

    def __init__(self, cv_folds: int = 5, logger: Optional[logging.Logger] = None):
        """
        Initialize model evaluator.

        Args:
            cv_folds: Number of folds for cross-validation
            logger: Optional logger instance
        """
        self.cv_folds = cv_folds
        self.logger = logger or logging.getLogger(__name__)

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        use_cross_validation: bool = False,
        return_predictions: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a model on test data.

        Args:
            model: Model with predict method
            X: Test features
            y: True target values
            use_cross_validation: Whether to perform cross-validation
            return_predictions: Whether to return predictions

        Returns:
            EvaluationResult with metrics and analysis
        """
        import time

        start_time = time.time()

        self.logger.info(f"Evaluating model on {len(X)} samples")

        # Make predictions
        if hasattr(model, "predict"):
            predictions = model.predict(X)
        else:
            raise ValueError("Model must have a predict method")

        # Handle different prediction formats
        if hasattr(predictions, "predicted_latency_ms"):
            predictions = np.array([p.predicted_latency_ms for p in predictions])
        elif hasattr(predictions, "predicted_jitter_ms"):
            predictions = np.array([p.predicted_jitter_ms for p in predictions])
        elif isinstance(predictions, np.ndarray):
            predictions = predictions.flatten()
        else:
            predictions = np.array(predictions).flatten()

        # Calculate errors
        errors = y - predictions

        # Calculate metrics
        metrics = EvaluationMetrics(
            r2_score=float(r2_score(y, predictions)),
            mae=float(mean_absolute_error(y, predictions)),
            rmse=float(np.sqrt(mean_squared_error(y, predictions))),
            mape=float(mean_absolute_percentage_error(y, predictions)),
            mean_error=float(np.mean(errors)),
            std_error=float(np.std(errors)),
            max_error=float(np.max(np.abs(errors))),
            min_error=float(np.min(np.abs(errors))),
        )

        # Cross-validation
        cv_scores = None
        if use_cross_validation and hasattr(model, "fit"):
            try:
                kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X, y, cv=kfold, scoring="r2", n_jobs=-1).tolist()
                self.logger.info(
                    f"Cross-validation R²: {np.mean(cv_scores):.4f} " f"(±{np.std(cv_scores):.4f})"
                )
            except Exception as e:
                self.logger.warning(f"Cross-validation failed: {e}")

        # Feature importance
        feature_importance = None
        if hasattr(model, "get_feature_importance"):
            try:
                feature_importance = model.get_feature_importance()
            except Exception:
                pass

        evaluation_time_ms = (time.time() - start_time) * 1000

        self.logger.info(
            f"Evaluation completed: R²={metrics.r2_score:.4f}, "
            f"MAE={metrics.mae:.2f}, RMSE={metrics.rmse:.2f}, "
            f"MAPE={metrics.mape:.2f}%"
        )

        return EvaluationResult(
            metrics=metrics,
            predictions=predictions if return_predictions else None,
            actual=y,
            errors=errors,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores,
            evaluation_time_ms=evaluation_time_ms,
        )

    def compare_models(
        self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray
    ) -> Dict[str, EvaluationResult]:
        """
        Compare multiple models.

        Args:
            models: Dictionary mapping model names to model objects
            X: Test features
            y: True target values

        Returns:
            Dictionary mapping model names to evaluation results
        """
        results = {}

        self.logger.info(f"Comparing {len(models)} models")

        for name, model in models.items():
            self.logger.info(f"Evaluating model: {name}")
            try:
                results[name] = self.evaluate(model, X, y)
            except Exception as e:
                self.logger.error(f"Error evaluating {name}: {e}")
                results[name] = None

        return results

    def generate_report(self, result: EvaluationResult, model_name: str = "Model") -> str:
        """
        Generate a text report from evaluation result.

        Args:
            result: Evaluation result
            model_name: Name of the model

        Returns:
            Formatted report string
        """
        report = f"""
{model_name} Evaluation Report
{'=' * 50}

Metrics:
  R² Score:        {result.metrics.r2_score:.4f}
  MAE:             {result.metrics.mae:.4f}
  RMSE:            {result.metrics.rmse:.4f}
  MAPE:            {result.metrics.mape:.2f}%

Error Statistics:
  Mean Error:      {result.metrics.mean_error:.4f}
  Std Error:       {result.metrics.std_error:.4f}
  Max Error:       {result.metrics.max_error:.4f}
  Min Error:       {result.metrics.min_error:.4f}

Evaluation Time:  {result.evaluation_time_ms:.2f} ms
"""

        if result.cross_validation_scores:
            cv_mean = np.mean(result.cross_validation_scores)
            cv_std = np.std(result.cross_validation_scores)
            report += f"""
Cross-Validation:
  Mean R²:         {cv_mean:.4f}
  Std R²:          {cv_std:.4f}
  Folds:           {len(result.cross_validation_scores)}
"""

        if result.feature_importance:
            report += "\nTop Features:\n"
            sorted_features = sorted(
                result.feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for name, importance in sorted_features:
                report += f"  {name}: {importance:.4f}\n"

        return report
