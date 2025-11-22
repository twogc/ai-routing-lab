"""Unit tests for ModelRegistry."""

from pathlib import Path

import pytest

from models.core.model_registry import ModelRegistry


@pytest.mark.unit
class TestModelRegistry:
    """Test suite for ModelRegistry."""

    def test_initialization(self, temp_models_dir):
        """Test registry initialization."""
        registry = ModelRegistry(models_dir=str(temp_models_dir))

        assert registry.models_dir == Path(temp_models_dir)
        assert registry.cache_size_mb == 1024
        assert len(registry.list_models()) == 0

    def test_register_model(self, temp_models_dir):
        """Test model registration."""
        registry = ModelRegistry(models_dir=str(temp_models_dir))

        # Create a simple model
        model = {"type": "test_model", "version": "1.0"}

        registry.register_model(
            model_id="test_model",
            model=model,
            model_type="prediction",
            accuracy=0.95,
            framework="test",
        )

        models = registry.list_models()
        assert any(m.model_id == "test_model" for m in models)
        metadata = registry.get_metadata("test_model")
        assert metadata.model_type == "prediction"
        assert metadata.accuracy == 0.95

    def test_get_model(self, temp_models_dir):
        """Test model retrieval."""
        registry = ModelRegistry(models_dir=str(temp_models_dir))

        # Register a model
        model = {"data": "test"}
        registry.register_model(
            model_id="test_model",
            model=model,
            model_type="prediction",
            accuracy=0.9,
            framework="test",
        )

        # Retrieve model
        retrieved_model, metadata = registry.get_model("test_model")

        assert retrieved_model == model
        assert metadata.model_id == "test_model"

    def test_get_nonexistent_model(self, temp_models_dir):
        """Test retrieving non-existent model raises error."""
        registry = ModelRegistry(models_dir=str(temp_models_dir))

        with pytest.raises(KeyError):
            registry.get_model("nonexistent")

    def test_list_models(self, temp_models_dir):
        """Test listing all models."""
        registry = ModelRegistry(models_dir=str(temp_models_dir))

        # Register multiple models
        for i in range(3):
            registry.register_model(
                model_id=f"model_{i}",
                model={"id": i},
                model_type="prediction",
                accuracy=0.9,
                framework="test",
            )

        models = registry.list_models()
        assert len(models) == 3

    def test_cache_functionality(self, temp_models_dir):
        """Test model caching."""
        registry = ModelRegistry(models_dir=str(temp_models_dir))

        model = {"data": "test"}
        registry.register_model(
            model_id="cached_model",
            model=model,
            model_type="prediction",
            accuracy=0.9,
            framework="test",
        )

        # First retrieval
        model1, _ = registry.get_model("cached_model")

        # Second retrieval (should be from cache)
        model2, _ = registry.get_model("cached_model")

        assert model1 == model2

        stats = registry.get_cache_stats()
        assert stats["cached_models"] == 1

    def test_clear_cache(self, temp_models_dir):
        """Test cache clearing."""
        registry = ModelRegistry(models_dir=str(temp_models_dir))

        model = {"data": "test"}
        registry.register_model(
            model_id="test_model",
            model=model,
            model_type="prediction",
            accuracy=0.9,
            framework="test",
        )

        # Load model to cache
        registry.get_model("test_model")

        # Clear cache
        registry.clear_cache()

        stats = registry.get_cache_stats()
        assert stats["cached_models"] == 0
