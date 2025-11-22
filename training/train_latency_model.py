"""
Training script for latency prediction model.

Trains a LatencyPredictor model on collected data and saves it to the model registry.
"""

import json
import logging
from pathlib import Path

import click
import numpy as np
from sklearn.model_selection import train_test_split

from models.core.model_registry import ModelRegistry
from models.prediction.latency_predictor import LatencyPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(data_path: Path):
    """
    Load training data from file.

    Args:
        data_path: Path to training data file (JSON or CSV)

    Returns:
        Tuple of (X, y) numpy arrays
    """
    logger.info(f"Loading training data from {data_path}")

    if data_path.suffix == ".json":
        with open(data_path, "r") as f:
            data = json.load(f)

        # Extract features and targets
        # Assuming data format: [{"features": [...], "latency": float}, ...]
        X = np.array([item["features"] for item in data])
        y = np.array([item["latency"] for item in data])

    elif data_path.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(data_path)

        # Assuming last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    return X, y


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to training data file",
)
@click.option(
    "--model-output",
    type=click.Path(path_type=Path),
    default="models/",
    help="Directory to save trained model",
)
@click.option("--n-estimators", type=int, default=100, help="Number of trees in Random Forest")
@click.option("--max-depth", type=int, default=15, help="Maximum depth of trees")
@click.option("--test-size", type=float, default=0.2, help="Fraction of data to use for testing")
@click.option("--random-state", type=int, default=42, help="Random seed for reproducibility")
@click.option("--use-ensemble", is_flag=True, help="Use ensemble with gradient boosting")
def train(
    data_path: Path,
    model_output: Path,
    n_estimators: int,
    max_depth: int,
    test_size: float,
    random_state: int,
    use_ensemble: bool,
):
    """Train latency prediction model."""

    logger.info("=" * 60)
    logger.info("Training Latency Prediction Model")
    logger.info("=" * 60)

    # Load data
    X, y = load_training_data(data_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Initialize model
    model = LatencyPredictor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        use_gradient_boosting=use_ensemble,
    )

    # Train model
    logger.info("Training model...")
    model.fit(X_train, y_train, use_ensemble=use_ensemble)

    # Evaluate model
    logger.info("Evaluating model...")
    metrics = model.evaluate(X_test, y_test, use_ensemble=use_ensemble)

    logger.info("=" * 60)
    logger.info("Model Performance:")
    logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f} ms")
    logger.info(f"  RMSE: {metrics['rmse']:.4f} ms")
    logger.info(f"  MAPE: {metrics['mape']:.4f}%")
    logger.info("=" * 60)

    # Check if model meets accuracy target
    if metrics["r2_score"] < 0.92:
        logger.warning(f"⚠️  Model accuracy ({metrics['r2_score']:.4f}) is below target (0.92)")
    else:
        logger.info(f"✅ Model accuracy ({metrics['r2_score']:.4f}) meets target!")

    # Save model to registry
    logger.info("Saving model to registry...")
    model_output.mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry(models_dir=str(model_output))
    registry.register_model(
        model_id="latency_predictor",
        model=model,
        model_type="prediction",
        accuracy=metrics["r2_score"],
        framework="scikit-learn",
    )

    logger.info(f"✅ Model saved successfully to {model_output}")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    train()
