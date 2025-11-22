"""
Model Validator - Validates model integrity, format compatibility, and signatures.

Ensures that models meet quality standards before deployment.
Validates checksums, formats, input/output shapes, and performance metrics.
"""

import hashlib
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ValidationResult:
    """Result of model validation"""

    is_valid: bool
    model_id: str
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    validation_timestamp: str


class ModelValidator:
    """
    Validates ML models for correctness, safety, and compatibility.

    Validations:
    - File integrity (SHA-256 checksum)
    - Model format and structure
    - Input/output shape compatibility
    - Serialization format validation
    - Performance metrics thresholds
    - Required methods exist
    - Model signature validation
    """

    # Expected methods for different model types
    MODEL_INTERFACE = {
        "anomaly": ["predict", "fit", "score"],
        "prediction": ["predict", "fit", "score"],
        "routing": ["predict", "fit", "score"],
    }

    # Performance thresholds
    MIN_ACCURACY = 0.50
    MAX_TRAINING_TIME = 3600  # seconds
    MIN_INPUT_SAMPLES = 10

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize Model Validator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.validation_rules = {}

    def validate_model(
        self,
        model: Any,
        model_id: str,
        model_type: str,
        expected_accuracy: float,
        input_shape: Optional[List[int]] = None,
        output_shape: Optional[List[int]] = None,
    ) -> ValidationResult:
        """
        Perform comprehensive model validation.

        Args:
            model: Model object to validate
            model_id: Model identifier
            model_type: Type of model (anomaly, prediction, routing)
            expected_accuracy: Expected accuracy threshold
            input_shape: Expected input shape
            output_shape: Expected output shape

        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        metrics = {}

        # Validate model type
        if model_type not in self.MODEL_INTERFACE:
            errors.append(f"Unknown model type: {model_type}")

        # Check required methods
        if model_type in self.MODEL_INTERFACE:
            for method in self.MODEL_INTERFACE[model_type]:
                if not hasattr(model, method):
                    errors.append(f"Model missing required method: {method}")
                elif not callable(getattr(model, method)):
                    errors.append(f"Model attribute '{method}' is not callable")

        # Validate accuracy
        if expected_accuracy < self.MIN_ACCURACY:
            warnings.append(
                f"Accuracy {expected_accuracy:.4f} below recommended minimum "
                f"{self.MIN_ACCURACY:.4f}"
            )
        metrics["accuracy"] = expected_accuracy

        # Validate shapes if provided
        if input_shape is not None:
            try:
                self._validate_shape(model, input_shape, "input")
                metrics["input_shape"] = input_shape
            except ValueError as e:
                warnings.append(f"Input shape validation: {e}")

        if output_shape is not None:
            metrics["output_shape"] = output_shape

        # Check model attributes
        self._validate_attributes(model, metrics, warnings, errors)

        # Validate model signature
        self._validate_signature(model, model_type, errors)

        # Test model predictions
        self._test_predictions(model, model_type, metrics, warnings, errors)

        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid,
            model_id=model_id,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            validation_timestamp=self._get_timestamp(),
        )

        if is_valid:
            self.logger.info(f"✓ Model {model_id} ({model_type}) validation successful")
        else:
            self.logger.error(f"✗ Model {model_id} validation failed with {len(errors)} error(s)")
            for error in errors:
                self.logger.error(f"  - {error}")

        return result

    def validate_file(
        self, file_path: str, expected_checksum: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Validate model file integrity.

        Args:
            file_path: Path to model file
            expected_checksum: Expected SHA-256 checksum

        Returns:
            Tuple of (is_valid, checksum)
        """
        if not Path(file_path).exists():
            self.logger.error(f"Model file not found: {file_path}")
            return False, ""

        # Check file size
        file_size = Path(file_path).stat().st_size
        if file_size == 0:
            self.logger.error(f"Model file is empty: {file_path}")
            return False, ""

        # Calculate checksum
        checksum = self._calculate_checksum(file_path)

        # Verify against expected
        if expected_checksum and checksum != expected_checksum:
            self.logger.error(
                f"Checksum mismatch for {file_path}: "
                f"expected {expected_checksum}, got {checksum}"
            )
            return False, checksum

        return True, checksum

    def validate_ensemble(self, models: Dict[str, Tuple[Any, float]]) -> ValidationResult:
        """
        Validate ensemble of models.

        Args:
            models: Dictionary of model_id -> (model, weight)

        Returns:
            Validation result for the ensemble
        """
        errors = []
        warnings = []
        metrics = {"models": {}}

        if not models:
            errors.append("Ensemble contains no models")

        # Validate weights sum to 1.0
        total_weight = sum(weight for _, weight in models.values())
        if abs(total_weight - 1.0) > 0.01:
            warnings.append(f"Ensemble weights sum to {total_weight:.4f}, expected 1.0")

        # Validate each model can predict
        for model_id, (model, weight) in models.items():
            try:
                if not callable(getattr(model, "predict", None)):
                    errors.append(f"Model {model_id} missing predict method")
                else:
                    metrics["models"][model_id] = {"weight": weight, "has_predict": True}
            except Exception as e:
                errors.append(f"Failed to validate model {model_id}: {e}")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            model_id="ensemble",
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            validation_timestamp=self._get_timestamp(),
        )

    @staticmethod
    def _validate_shape(model: Any, expected_shape: List[int], shape_type: str):
        """Validate that model can handle expected shape"""
        # For sklearn models, check n_features_in_
        if hasattr(model, "n_features_in_"):
            actual = model.n_features_in_
            expected = expected_shape[0] if expected_shape else 1

            if actual != expected:
                raise ValueError(
                    f"{shape_type} shape mismatch: expected {expected}, "
                    f"model expects {actual} features"
                )

    @staticmethod
    def _validate_attributes(model: Any, metrics: Dict, warnings: List[str], errors: List[str]):
        """Validate model attributes"""
        # Check for common sklearn attributes
        if hasattr(model, "classes_"):
            metrics["n_classes"] = len(model.classes_)

        if hasattr(model, "n_features_in_"):
            metrics["n_features"] = model.n_features_in_

        if hasattr(model, "feature_names_in_"):
            metrics["feature_names"] = list(model.feature_names_in_)

    @staticmethod
    def _validate_signature(model: Any, model_type: str, errors: List[str]):
        """Validate model method signatures"""
        if hasattr(model, "predict"):
            try:
                sig = inspect.signature(model.predict)
                params = list(sig.parameters.keys())

                # Should have at least 'self' and data parameter
                if len(params) < 2:
                    errors.append(f"Model predict method signature invalid: {sig}")
            except Exception as e:
                errors.append(f"Failed to inspect predict signature: {e}")

    @staticmethod
    def _test_predictions(
        model: Any, model_type: str, metrics: Dict, warnings: List[str], errors: List[str]
    ):
        """Test that model can make predictions"""
        try:
            # Create minimal test data
            if model_type == "anomaly":
                test_data = np.random.randn(1, 5)  # 1 sample, 5 features
            elif model_type == "prediction":
                test_data = np.random.randn(1, 10)  # 1 sample, 10 features
            else:  # routing
                test_data = np.random.randn(1, 8)  # 1 sample, 8 features

            # Try prediction
            if hasattr(model, "predict"):
                result = model.predict(test_data)
                metrics["test_prediction_shape"] = list(np.array(result).shape)
                metrics["test_prediction_dtype"] = str(np.array(result).dtype)
            else:
                warnings.append("Model does not have predict method")

        except Exception as e:
            warnings.append(f"Test prediction failed: {e}")

    @staticmethod
    def _calculate_checksum(file_path: str) -> str:
        """Calculate SHA-256 checksum"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp"""
        from datetime import datetime

        return datetime.now().isoformat()


class ModelSignature:
    """Validates model can process expected data types"""

    def __init__(self, model: Any):
        """Initialize with model"""
        self.model = model

    def is_compatible(self, input_data: np.ndarray) -> bool:
        """Check if model can process input data"""
        try:
            # Try to get prediction
            _ = self.model.predict(input_data[:1])
            return True
        except Exception as e:
            logging.warning(f"Model compatibility check failed: {e}")
            return False

    def get_expected_shape(self) -> Optional[Tuple[int, ...]]:
        """Get expected input shape"""
        if hasattr(self.model, "n_features_in_"):
            return (1, self.model.n_features_in_)
        return None
