"""
Data Pipeline - Processes data through multiple stages.

Provides a pipeline framework for data transformation, cleaning, and preparation.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from models.core.data_preprocessor import DataPreprocessor
from models.core.feature_extractor import FeatureExtractor


@dataclass
class PipelineStage:
    """Represents a stage in the data pipeline."""

    name: str
    processor: Callable
    enabled: bool = True


class DataPipeline:
    """
    Pipeline for processing data through multiple stages.

    Stages can include:
    - Data cleaning
    - Feature extraction
    - Normalization
    - Validation
    """

    def __init__(
        self, stages: Optional[List[PipelineStage]] = None, logger: Optional[logging.Logger] = None
    ):
        """
        Initialize data pipeline.

        Args:
            stages: List of pipeline stages
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.stages = stages or []
        self.preprocessor: Optional[DataPreprocessor] = None
        self.feature_extractor: Optional[FeatureExtractor] = None

    def add_stage(self, stage: PipelineStage):
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        self.logger.info(f"Added pipeline stage: {stage.name}")

    def add_preprocessing_stage(
        self, strategy: str = "mean", outlier_method: str = "iqr", normalization: str = "standard"
    ):
        """Add data preprocessing stage."""
        self.preprocessor = DataPreprocessor(
            strategy=strategy,
            outlier_method=outlier_method,
            normalization=normalization,
            logger=self.logger,
        )
        stage = PipelineStage(
            name="preprocessing", processor=self.preprocessor.fit_transform, enabled=True
        )
        self.add_stage(stage)

    def add_feature_extraction_stage(
        self,
        window_sizes: Optional[List[int]] = None,
        include_stats: bool = True,
        include_rolling: bool = True,
        include_ema: bool = True,
    ):
        """Add feature extraction stage."""
        self.feature_extractor = FeatureExtractor(window_sizes=window_sizes, logger=self.logger)

        def extract_features(X, **kwargs):
            result = self.feature_extractor.extract_all_features(
                X,
                include_stats=include_stats,
                include_rolling=include_rolling,
                include_ema=include_ema,
            )
            return result.features, result

        stage = PipelineStage(name="feature_extraction", processor=extract_features, enabled=True)
        self.add_stage(stage)

    def process(
        self, X: np.ndarray, feature_names: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Process data through all enabled stages.

        Args:
            X: Input data
            feature_names: Optional feature names
            **kwargs: Additional arguments for stages

        Returns:
            Dictionary with processed data and metadata
        """
        start_time = time.time()

        self.logger.info(f"Processing data through {len(self.stages)} stages")

        current_data = X
        metadata = {"original_shape": X.shape, "stages_applied": [], "processing_time_ms": 0.0}

        for stage in self.stages:
            if not stage.enabled:
                self.logger.debug(f"Skipping disabled stage: {stage.name}")
                continue

            try:
                self.logger.debug(f"Applying stage: {stage.name}")
                stage_start = time.time()

                # Apply stage
                if stage.name == "preprocessing" and self.preprocessor:
                    current_data, stats = stage.processor(
                        current_data, feature_names=feature_names, **kwargs
                    )
                    metadata["preprocessing_stats"] = stats
                elif stage.name == "feature_extraction" and self.feature_extractor:
                    current_data, extraction_result = stage.processor(current_data, **kwargs)
                    metadata["feature_extraction"] = {
                        "n_features": extraction_result.n_features_created,
                        "feature_names": extraction_result.feature_names,
                        "extraction_time_ms": extraction_result.extraction_time_ms,
                    }
                else:
                    current_data = stage.processor(current_data, **kwargs)

                stage_time = (time.time() - stage_start) * 1000
                metadata["stages_applied"].append({"name": stage.name, "time_ms": stage_time})

                self.logger.debug(
                    f"Stage {stage.name} completed in {stage_time:.2f}ms, "
                    f"output shape: {current_data.shape}"
                )

            except Exception as e:
                self.logger.error(f"Error in stage {stage.name}: {e}")
                raise

        metadata["final_shape"] = current_data.shape
        metadata["processing_time_ms"] = (time.time() - start_time) * 1000

        self.logger.info(
            f"Pipeline processing completed in {metadata['processing_time_ms']:.2f}ms, "
            f"shape: {metadata['original_shape']} -> {metadata['final_shape']}"
        )

        return {"data": current_data, "metadata": metadata}

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit pipeline on data (for stages that require fitting).

        Args:
            X: Training data
            feature_names: Optional feature names
        """
        if self.preprocessor:
            self.preprocessor.fit(X, feature_names=feature_names)
            self.logger.info("Preprocessor fitted")

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Transform data using fitted pipeline.

        Args:
            X: Input data
            **kwargs: Additional arguments

        Returns:
            Transformed data
        """
        result = self.process(X, **kwargs)
        return result["data"]

    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get stage by name."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def enable_stage(self, name: str):
        """Enable a stage by name."""
        stage = self.get_stage(name)
        if stage:
            stage.enabled = True
            self.logger.info(f"Enabled stage: {name}")
        else:
            self.logger.warning(f"Stage not found: {name}")

    def disable_stage(self, name: str):
        """Disable a stage by name."""
        stage = self.get_stage(name)
        if stage:
            stage.enabled = False
            self.logger.info(f"Disabled stage: {name}")
        else:
            self.logger.warning(f"Stage not found: {name}")
