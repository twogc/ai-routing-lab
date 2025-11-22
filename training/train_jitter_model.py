"""
Training script for jitter prediction model.

Trains a JitterPredictor model on collected data and saves it to the model registry.
"""

import json
import logging
from pathlib import Path

import click
import numpy as np
from sklearn.model_selection import train_test_split

from models.core.model_registry import ModelRegistry
from models.prediction.jitter_predictor import JitterPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(data_path: Path):
    """Load training data from file."""
    logger.info(f"Loading training data from {data_path}")

    if data_path.suffix == ".json":
        with open(data_path, "r") as f:
            data = json.load(f)

        X = np.array([item["features"] for item in data])
        y = np.array([item["jitter"] for item in data])

    elif data_path.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    return X, y


@click.command()
@click.option("--data-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model-output", type=click.Path(path_type=Path), default="models/")
@click.option("--n-estimators", type=int, default=100)
@click.option("--max-depth", type=int, default=15)
@click.option("--test-size", type=float, default=0.2)
@click.option("--random-state", type=int, default=42)
def train(
    data_path: Path,
    model_output: Path,
    n_estimators: int,
    max_depth: int,
    test_size: float,
    random_state: int,
):
    """Train jitter prediction model."""

    logger.info("=" * 60)
    logger.info("Training Jitter Prediction Model")
    logger.info("=" * 60)

    # Load and split data
    X, y = load_training_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Train model
    model = JitterPredictor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )

    logger.info("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    logger.info("Evaluating model...")
    metrics = model.evaluate(X_test, y_test)

    logger.info("=" * 60)
    logger.info("Model Performance:")
    logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f} ms")
    logger.info(f"  RMSE: {metrics['rmse']:.4f} ms")
    logger.info("=" * 60)

    if metrics["r2_score"] < 0.92:
        logger.warning("⚠️  Model accuracy below target (0.92)")
    else:
        logger.info("✅ Model accuracy meets target!")

    # Save model
    logger.info("Saving model...")
    model_output.mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry(models_dir=str(model_output))
    registry.register_model(
        model_id="jitter_predictor",
        model=model,
        model_type="prediction",
        accuracy=metrics["r2_score"],
        framework="scikit-learn",
    )

    logger.info(f"✅ Model saved to {model_output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
