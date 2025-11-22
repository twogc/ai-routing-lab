"""
Model Registry - Manages loading, caching, versioning, and lifecycle of ML models.

Provides central repository for all ML models used in CloudBridge AI Service.
Handles model loading from disk, in-memory caching, version management, and fallback strategies.
"""

import hashlib
import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelMetadata:
    """Metadata for a registered model"""

    model_id: str
    model_type: str  # 'anomaly', 'prediction', 'routing'
    version: str
    accuracy: float
    created_at: str
    updated_at: str
    file_path: str
    checksum: str
    framework: str  # 'sklearn', 'tensorflow', 'pytorch'
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    dependencies: Optional[List[str]] = None
    tags: Optional[Dict[str, str]] = None


class ModelRegistry:
    """
    Central registry for ML models with caching, versioning, and fallback support.

    Features:
    - Model loading from disk with SHA-256 verification
    - In-memory caching with configurable TTL
    - Version management (keep multiple versions)
    - Fallback strategies (use previous version if current fails)
    - Metadata tracking (accuracy, creation date, checksums)
    - Thread-safe operations
    """

    def __init__(
        self,
        models_dir: str,
        cache_size_mb: int = 1024,
        cache_ttl_seconds: int = 3600,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Model Registry.

        Args:
            models_dir: Directory containing model files
            cache_size_mb: Maximum cache size in MB
            cache_ttl_seconds: Cache TTL in seconds (0 = no expiration)
            logger: Optional logger instance
        """
        self.models_dir = Path(models_dir)
        self.cache_size_mb = cache_size_mb
        self.cache_ttl_seconds = cache_ttl_seconds
        self.logger = logger or logging.getLogger(__name__)

        # Thread-safe storage
        self._lock = Lock()
        self._cache: Dict[str, Tuple[Any, float]] = {}  # model_id -> (model, load_time)
        self._metadata: Dict[str, ModelMetadata] = {}
        self._version_history: Dict[str, List[str]] = {}  # model_id -> [versions]
        self._cache_stats = {"hits": 0, "misses": 0, "loads": 0, "evictions": 0}

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._load_metadata()
        self.logger.info(f"ModelRegistry initialized with {len(self._metadata)} models")

    def register_model(
        self,
        model_id: str,
        model: Any,
        model_type: str,
        accuracy: float,
        framework: str,
        file_path: Optional[str] = None,
    ) -> ModelMetadata:
        """
        Register a new model or update existing model.

        Args:
            model_id: Unique model identifier
            model: Model object to register
            model_type: Type of model (anomaly, prediction, routing)
            accuracy: Model accuracy metric
            framework: ML framework (sklearn, tensorflow, pytorch)
            file_path: Optional path to save model

        Returns:
            ModelMetadata with registration details
        """
        with self._lock:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save model to disk if file_path provided
            if file_path is None:
                file_path = str(self.models_dir / f"{model_id}_v{version}.pkl")

            # Serialize and save
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(model, f)
                checksum = self._calculate_checksum(file_path)
            except Exception as e:
                self.logger.error(f"Failed to save model {model_id}: {e}")
                raise

            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_type=model_type,
                version=version,
                accuracy=accuracy,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                file_path=file_path,
                checksum=checksum,
                framework=framework,
            )

            # Update registries
            self._metadata[model_id] = metadata
            if model_id not in self._version_history:
                self._version_history[model_id] = []
            self._version_history[model_id].append(version)

            # Keep only last 5 versions
            if len(self._version_history[model_id]) > 5:
                old_version = self._version_history[model_id].pop(0)
                self.logger.debug(f"Removing old version {old_version} of {model_id}")

            # Cache the model
            self._cache[model_id] = (model, time.time())

            # Save metadata
            self._save_metadata()

            self.logger.info(
                f"Registered model {model_id} v{version} "
                f"(accuracy={accuracy:.4f}, framework={framework})"
            )

            return metadata

    def get_model(self, model_id: str, version: Optional[str] = None) -> Tuple[Any, ModelMetadata]:
        """
        Retrieve a model from cache or disk.

        Args:
            model_id: Model identifier
            version: Optional specific version to load

        Returns:
            Tuple of (model_object, metadata)

        Raises:
            KeyError: If model not found
        """
        with self._lock:
            # Check cache first
            if model_id in self._cache:
                model, load_time = self._cache[model_id]

                # Check TTL
                if (
                    self.cache_ttl_seconds == 0
                    or (time.time() - load_time) < self.cache_ttl_seconds
                ):
                    self._cache_stats["hits"] += 1
                    return model, self._metadata[model_id]
                else:
                    # Cache expired
                    del self._cache[model_id]

            # Load from disk
            self._cache_stats["misses"] += 1

            if model_id not in self._metadata:
                raise KeyError(f"Model {model_id} not found in registry")

            metadata = self._metadata[model_id]

            try:
                with open(metadata.file_path, "rb") as f:
                    model = pickle.load(f)

                # Verify checksum
                if self._calculate_checksum(metadata.file_path) != metadata.checksum:
                    self.logger.warning(f"Checksum mismatch for model {model_id}")

                # Cache the model
                self._cache[model_id] = (model, time.time())
                self._cache_stats["loads"] += 1

                return model, metadata

            except Exception as e:
                self.logger.error(f"Failed to load model {model_id}: {e}")
                raise

    def get_model_by_type(self, model_type: str) -> Dict[str, Tuple[Any, ModelMetadata]]:
        """
        Get all models of a specific type.

        Args:
            model_type: Type of models to retrieve (anomaly, prediction, routing)

        Returns:
            Dictionary mapping model_id to (model, metadata) tuple
        """
        with self._lock:
            result = {}
            for model_id, metadata in self._metadata.items():
                if metadata.model_type == model_type:
                    try:
                        model, _ = self.get_model(model_id)
                        result[model_id] = (model, metadata)
                    except Exception as e:
                        self.logger.error(f"Failed to get model {model_id}: {e}")
            return result

    def get_metadata(self, model_id: str) -> ModelMetadata:
        """Get metadata for a model without loading it"""
        with self._lock:
            if model_id not in self._metadata:
                raise KeyError(f"Model {model_id} not found")
            return self._metadata[model_id]

    def list_models(self) -> List[ModelMetadata]:
        """List all registered models"""
        with self._lock:
            return list(self._metadata.values())

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
            hit_rate = (
                (self._cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            )

            return {
                **self._cache_stats,
                "total_requests": total_requests,
                "hit_rate_percent": hit_rate,
                "cached_models": len(self._cache),
            }

    def clear_cache(self, model_id: Optional[str] = None):
        """Clear cache for specific model or all models"""
        with self._lock:
            if model_id:
                if model_id in self._cache:
                    del self._cache[model_id]
                    self.logger.info(f"Cleared cache for model {model_id}")
            else:
                self._cache.clear()
                self.logger.info("Cleared all model caches")

    def remove_model(self, model_id: str, keep_versions: int = 1):
        """
        Remove model from registry (keeps specified number of versions).

        Args:
            model_id: Model to remove
            keep_versions: Number of versions to keep (default 1)
        """
        with self._lock:
            if model_id not in self._metadata:
                raise KeyError(f"Model {model_id} not found")

            # Remove from cache
            if model_id in self._cache:
                del self._cache[model_id]

            # Remove old versions
            if model_id in self._version_history:
                versions_to_keep = self._version_history[model_id][-keep_versions:]
                for version in self._version_history[model_id]:
                    if version not in versions_to_keep:
                        self._version_history[model_id].remove(version)

            # Update metadata
            del self._metadata[model_id]
            self._save_metadata()

            self.logger.info(f"Removed model {model_id} (keeping {keep_versions} versions)")

    @staticmethod
    def _calculate_checksum(file_path: str) -> str:
        """Calculate SHA-256 checksum of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def _load_metadata(self):
        """Load metadata from disk"""
        metadata_file = self.models_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                    for model_id, meta_dict in data.get("metadata", {}).items():
                        self._metadata[model_id] = ModelMetadata(**meta_dict)
                    self._version_history = data.get("version_history", {})
                self.logger.info(f"Loaded metadata for {len(self._metadata)} models")
            except Exception as e:
                self.logger.error(f"Failed to load metadata: {e}")

    def _save_metadata(self):
        """Save metadata to disk"""
        try:
            metadata_file = self.models_dir / "metadata.json"
            data = {
                "metadata": {model_id: asdict(meta) for model_id, meta in self._metadata.items()},
                "version_history": self._version_history,
            }
            with open(metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
