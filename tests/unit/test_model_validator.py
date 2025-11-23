"""Unit tests for ModelValidator."""

import pytest
import tempfile
import os
import numpy as np
from unittest.mock import MagicMock, patch
from models.core.model_validator import ModelValidator, ValidationResult, ModelSignature

class TestModelValidator:
    """Test suite for ModelValidator."""

    @pytest.fixture
    def validator(self):
        return ModelValidator()

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.predict = MagicMock(return_value=np.array([1.0]))
        model.fit = MagicMock()
        model.score = MagicMock()
        return model

    def test_validate_model_valid(self, validator, mock_model):
        """Test validation of a valid model."""
        result = validator.validate_model(
            model=mock_model,
            model_id="test_model",
            model_type="prediction",
            expected_accuracy=0.95
        )
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.metrics["accuracy"] == 0.95

    def test_validate_model_invalid_type(self, validator, mock_model):
        """Test validation with invalid model type."""
        result = validator.validate_model(
            model=mock_model,
            model_id="test_model",
            model_type="unknown_type",
            expected_accuracy=0.95
        )
        
        assert not result.is_valid
        assert any("Unknown model type" in err for err in result.errors)

    def test_validate_model_missing_methods(self, validator):
        """Test validation with missing required methods."""
        class IncompleteModel:
            pass
            
        model = IncompleteModel()
        
        result = validator.validate_model(
            model=model,
            model_id="test_model",
            model_type="prediction",
            expected_accuracy=0.95
        )
        
        assert not result.is_valid
        assert any("missing required method" in err for err in result.errors)

    def test_validate_model_low_accuracy(self, validator, mock_model):
        """Test validation with low accuracy."""
        result = validator.validate_model(
            model=mock_model,
            model_id="test_model",
            model_type="prediction",
            expected_accuracy=0.1
        )
        
        assert result.is_valid  # Still valid, but with warning
        assert len(result.warnings) > 0
        assert any("below recommended minimum" in warn for warn in result.warnings)

    def test_validate_file(self, validator):
        """Test file validation."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f_path = f.name
            
        try:
            # Valid file
            is_valid, checksum = validator.validate_file(f_path)
            assert is_valid
            assert len(checksum) > 0
            
            # Invalid checksum
            is_valid, _ = validator.validate_file(f_path, expected_checksum="wrong_checksum")
            assert not is_valid
            
            # Missing file
            is_valid, _ = validator.validate_file("non_existent_file")
            assert not is_valid
            
        finally:
            os.unlink(f_path)

    def test_validate_ensemble(self, validator, mock_model):
        """Test ensemble validation."""
        models = {
            "model1": (mock_model, 0.6),
            "model2": (mock_model, 0.4)
        }
        
        result = validator.validate_ensemble(models)
        
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_ensemble_invalid_weights(self, validator, mock_model):
        """Test ensemble validation with invalid weights."""
        models = {
            "model1": (mock_model, 0.6),
            "model2": (mock_model, 0.6)  # Sum > 1.0
        }
        
        result = validator.validate_ensemble(models)
        
        assert result.is_valid  # Warning only
        assert len(result.warnings) > 0
        assert any("weights sum to" in warn for warn in result.warnings)

    def test_validate_shape(self, validator, mock_model):
        """Test shape validation."""
        mock_model.n_features_in_ = 10
        
        # Correct shape
        validator._validate_shape(mock_model, [10], "input")
        
        # Incorrect shape
        with pytest.raises(ValueError):
            validator._validate_shape(mock_model, [5], "input")

class TestModelSignature:
    """Test suite for ModelSignature."""

    def test_is_compatible(self):
        """Test compatibility check."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        
        sig = ModelSignature(mock_model)
        assert sig.is_compatible(np.array([[1, 2, 3]]))
        
        mock_model.predict.side_effect = Exception("Error")
        assert not sig.is_compatible(np.array([[1, 2, 3]]))

    def test_get_expected_shape(self):
        """Test getting expected shape."""
        mock_model = MagicMock()
        mock_model.n_features_in_ = 10
        
        sig = ModelSignature(mock_model)
        assert sig.get_expected_shape() == (1, 10)
        
        del mock_model.n_features_in_
        assert sig.get_expected_shape() is None
