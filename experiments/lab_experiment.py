"""
Laboratory Experiment Framework for AI Routing Lab

Provides infrastructure for creating, running, and tracking ML experiments.
Based on CloudBridge AI Service ecosystem patterns.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.core.model_registry import ModelRegistry, ModelMetadata
from models.core.data_preprocessor import DataPreprocessor
from models.core.feature_extractor import FeatureExtractor


logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a laboratory experiment"""
    experiment_id: str
    experiment_name: str
    description: str
    model_type: str  # 'latency', 'jitter', 'route_selection'
    model_framework: str  # 'tensorflow', 'pytorch', 'sklearn'
    hyperparameters: Dict[str, Any]
    data_config: Dict[str, Any]
    evaluation_metrics: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class ExperimentResult:
    """Results from a laboratory experiment"""
    experiment_id: str
    model_id: str
    accuracy: float
    metrics: Dict[str, float]
    training_time_seconds: float
    inference_time_ms: float
    model_size_mb: float
    completed_at: str
    artifacts: Dict[str, str]  # Paths to saved artifacts


class LaboratoryExperiment:
    """
    Framework for creating and running ML laboratory experiments.
    
    Features:
    - Experiment configuration and tracking
    - Model training and evaluation
    - Results storage and comparison
    - Integration with Model Registry
    - Reproducibility support
    """
    
    def __init__(
        self,
        experiments_dir: str = "experiments/results",
        models_dir: str = "models",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Laboratory Experiment framework.
        
        Args:
            experiments_dir: Directory for experiment results
            models_dir: Directory for model storage
            logger: Optional logger instance
        """
        self.experiments_dir = Path(experiments_dir)
        self.models_dir = Path(models_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Create directories
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_registry = ModelRegistry(
            models_dir=str(self.models_dir),
            logger=self.logger
        )
        self.preprocessor = DataPreprocessor(logger=self.logger)
        self.feature_extractor = FeatureExtractor(logger=self.logger)
        
        # Experiment tracking
        self.current_experiment: Optional[ExperimentConfig] = None
        self.experiment_results: List[ExperimentResult] = []
    
    def create_experiment(
        self,
        experiment_name: str,
        description: str,
        model_type: str,
        model_framework: str,
        hyperparameters: Dict[str, Any],
        data_config: Dict[str, Any],
        evaluation_metrics: List[str]
    ) -> ExperimentConfig:
        """
        Create a new laboratory experiment.
        
        Args:
            experiment_name: Name of the experiment
            description: Description of what the experiment tests
            model_type: Type of model ('latency', 'jitter', 'route_selection')
            model_framework: ML framework to use
            hyperparameters: Model hyperparameters
            data_config: Data collection and preprocessing config
            evaluation_metrics: List of metrics to evaluate
            
        Returns:
            ExperimentConfig object
        """
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            description=description,
            model_type=model_type,
            model_framework=model_framework,
            hyperparameters=hyperparameters,
            data_config=data_config,
            evaluation_metrics=evaluation_metrics
        )
        
        self.current_experiment = config
        
        # Save experiment config
        config_file = self.experiments_dir / f"{experiment_id}_config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        self.logger.info(f"Created experiment: {experiment_id}")
        
        return config
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Prepare data for experiment using preprocessor and feature extractor.
        
        Args:
            X: Input features
            y: Optional target values
            timestamps: Optional timestamps for time features
            
        Returns:
            Tuple of (processed_X, processed_y, preprocessing_info)
        """
        if self.current_experiment is None:
            raise RuntimeError("No experiment created. Call create_experiment() first.")
        
        config = self.current_experiment.data_config
        
        # Feature extraction
        if config.get('extract_features', True):
            feature_result = self.feature_extractor.extract_all_features(
                X,
                timestamps=timestamps,
                include_stats=config.get('include_stats', True),
                include_rolling=config.get('include_rolling', True),
                include_ema=config.get('include_ema', True)
            )
            X = feature_result.features
            self.logger.info(f"Extracted {feature_result.n_features_created} features")
        
        # Data preprocessing
        X_processed, stats = self.preprocessor.fit_transform(
            X,
            remove_outliers=config.get('remove_outliers', True),
            normalize=config.get('normalize', True)
        )
        
        preprocessing_info = {
            'original_shape': stats.original_shape,
            'final_shape': stats.final_shape,
            'missing_values_handled': stats.missing_values_handled,
            'outliers_removed': stats.outliers_removed,
            'preprocessing_time_ms': stats.preprocessing_time_ms
        }
        
        return X_processed, y, preprocessing_info
    
    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a model for the experiment.
        
        Args:
            X: Training features
            y: Training targets
            model: Model object to train
            
        Returns:
            Tuple of (trained_model, training_info)
        """
        if self.current_experiment is None:
            raise RuntimeError("No experiment created. Call create_experiment() first.")
        
        start_time = time.time()
        
        # Train model
        if hasattr(model, 'fit'):
            model.fit(X, y)
        else:
            raise ValueError("Model must have a fit() method")
        
        training_time = time.time() - start_time
        
        training_info = {
            'training_time_seconds': training_time,
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }
        
        self.logger.info(f"Model trained in {training_time:.2f}s")
        
        return model, training_info
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.current_experiment is None:
            raise RuntimeError("No experiment created. Call create_experiment() first.")
        
        start_time = time.time()
        
        # Make predictions
        if hasattr(model, 'predict'):
            predictions = model.predict(X_test)
        else:
            raise ValueError("Model must have a predict() method")
        
        inference_time = (time.time() - start_time) * 1000 / len(X_test)
        
        # Calculate metrics
        metrics = {}
        
        for metric_name in self.current_experiment.evaluation_metrics:
            if metric_name == 'accuracy':
                if y_test.dtype == int or y_test.dtype == bool:
                    metrics['accuracy'] = float(np.mean(predictions == y_test))
                else:
                    # For regression, use R² score
                    from sklearn.metrics import r2_score
                    metrics['accuracy'] = float(r2_score(y_test, predictions))
            
            elif metric_name == 'mae':
                from sklearn.metrics import mean_absolute_error
                metrics['mae'] = float(mean_absolute_error(y_test, predictions))
            
            elif metric_name == 'rmse':
                from sklearn.metrics import mean_squared_error
                metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, predictions)))
            
            elif metric_name == 'mape':
                # Mean Absolute Percentage Error
                mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-8))) * 100
                metrics['mape'] = float(mape)
        
        metrics['inference_time_ms'] = inference_time
        
        return metrics
    
    def save_experiment_results(
        self,
        model: Any,
        metrics: Dict[str, float],
        training_info: Dict[str, Any],
        preprocessing_info: Dict[str, Any]
    ) -> ExperimentResult:
        """
        Save experiment results and register model.
        
        Args:
            model: Trained model
            metrics: Evaluation metrics
            training_info: Training information
            preprocessing_info: Preprocessing information
            
        Returns:
            ExperimentResult object
        """
        if self.current_experiment is None:
            raise RuntimeError("No experiment created. Call create_experiment() first.")
        
        # Register model
        model_id = f"{self.current_experiment.experiment_id}_model"
        accuracy = metrics.get('accuracy', metrics.get('r2_score', 0.0))
        
        metadata = self.model_registry.register_model(
            model_id=model_id,
            model=model,
            model_type=self.current_experiment.model_type,
            accuracy=accuracy,
            framework=self.current_experiment.model_framework
        )
        
        # Calculate model size
        import pickle
        import os
        model_bytes = len(pickle.dumps(model))
        model_size_mb = model_bytes / (1024 * 1024)
        
        # Create result
        result = ExperimentResult(
            experiment_id=self.current_experiment.experiment_id,
            model_id=model_id,
            accuracy=accuracy,
            metrics=metrics,
            training_time_seconds=training_info['training_time_seconds'],
            inference_time_ms=metrics.get('inference_time_ms', 0.0),
            model_size_mb=model_size_mb,
            completed_at=datetime.now().isoformat(),
            artifacts={
                'model_file': metadata.file_path,
                'config_file': str(self.experiments_dir / f"{self.current_experiment.experiment_id}_config.json")
            }
        )
        
        # Save results
        results_file = self.experiments_dir / f"{self.current_experiment.experiment_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        self.experiment_results.append(result)
        
        self.logger.info(
            f"Experiment {self.current_experiment.experiment_id} completed: "
            f"accuracy={accuracy:.4f}"
        )
        
        return result
    
    def compare_experiments(self) -> pd.DataFrame:
        """
        Compare results from multiple experiments.
        
        Returns:
            DataFrame with experiment comparison
        """
        if not self.experiment_results:
            return pd.DataFrame()
        
        comparison_data = []
        for result in self.experiment_results:
            comparison_data.append({
                'experiment_id': result.experiment_id,
                'model_id': result.model_id,
                'accuracy': result.accuracy,
                'training_time_seconds': result.training_time_seconds,
                'inference_time_ms': result.inference_time_ms,
                'model_size_mb': result.model_size_mb,
                **result.metrics
            })
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def list_experiments(self) -> List[ExperimentConfig]:
        """List all saved experiments"""
        experiments = []
        
        for config_file in self.experiments_dir.glob("*_config.json"):
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
                experiments.append(ExperimentConfig(**config_dict))
        
        return experiments


def create_latency_prediction_experiment(
    hyperparameters: Optional[Dict[str, Any]] = None,
    data_config: Optional[Dict[str, Any]] = None
) -> LaboratoryExperiment:
    """
    Create a standard latency prediction experiment.
    
    Args:
        hyperparameters: Optional model hyperparameters
        data_config: Optional data configuration
        
    Returns:
        LaboratoryExperiment instance
    """
    lab = LaboratoryExperiment()
    
    lab.create_experiment(
        experiment_name="latency_prediction",
        description="Predict route latency using ML models",
        model_type="latency",
        model_framework="sklearn",
        hyperparameters=hyperparameters or {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        data_config=data_config or {
            'extract_features': True,
            'include_stats': True,
            'include_rolling': True,
            'include_ema': True,
            'remove_outliers': True,
            'normalize': True
        },
        evaluation_metrics=['accuracy', 'mae', 'rmse', 'mape']
    )
    
    return lab


def create_jitter_prediction_experiment(
    hyperparameters: Optional[Dict[str, Any]] = None,
    data_config: Optional[Dict[str, Any]] = None
) -> LaboratoryExperiment:
    """
    Create a standard jitter prediction experiment.
    
    Args:
        hyperparameters: Optional model hyperparameters
        data_config: Optional data configuration
        
    Returns:
        LaboratoryExperiment instance
    """
    lab = LaboratoryExperiment()
    
    lab.create_experiment(
        experiment_name="jitter_prediction",
        description="Predict route jitter using ML models",
        model_type="jitter",
        model_framework="sklearn",
        hyperparameters=hyperparameters or {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        data_config=data_config or {
            'extract_features': True,
            'include_stats': True,
            'include_rolling': True,
            'include_ema': True,
            'remove_outliers': True,
            'normalize': True
        },
        evaluation_metrics=['accuracy', 'mae', 'rmse', 'mape']
    )
    
    return lab

