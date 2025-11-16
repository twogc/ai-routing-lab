"""
Example: Creating and running a laboratory experiment for latency prediction

This example demonstrates how to use the Laboratory Experiment framework
to create, run, and evaluate ML experiments for route latency prediction.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.lab_experiment import (
    LaboratoryExperiment,
    create_latency_prediction_experiment
)
from models.prediction import LatencyPredictor, JitterPredictor, RoutePredictionEnsemble


def generate_sample_data(n_samples: int = 1000) -> tuple:
    """
    Generate sample latency data for experimentation.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (X, y, timestamps)
    """
    np.random.seed(42)
    
    # Generate features: route characteristics, network conditions, etc.
    X = np.random.randn(n_samples, 8)
    
    # Generate target: latency (ms)
    # Latency depends on features with some noise
    y = (
        20 +  # Base latency
        5 * X[:, 0] +  # Route distance factor
        3 * X[:, 1] +  # Network congestion
        2 * X[:, 2] +  # Packet loss impact
        np.random.randn(n_samples) * 2  # Noise
    )
    y = np.maximum(y, 5)  # Minimum latency of 5ms
    
    # Generate timestamps
    timestamps = np.arange(n_samples)
    
    return X, y, timestamps


def run_latency_experiment():
    """Run a complete latency prediction experiment"""
    
    print("=" * 60)
    print("AI Routing Lab - Latency Prediction Experiment")
    print("=" * 60)
    
    # Step 1: Create experiment
    print("\n1. Creating experiment...")
    lab = create_latency_prediction_experiment(
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
    )
    
    # Step 2: Generate sample data
    print("\n2. Generating sample data...")
    X, y, timestamps = generate_sample_data(n_samples=1000)
    print(f"   Generated {len(X)} samples with {X.shape[1]} features")
    
    # Step 3: Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(
        X, y, timestamps, test_size=0.2, random_state=42
    )
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Step 4: Prepare data
    print("\n4. Preparing data (feature extraction + preprocessing)...")
    X_train_processed, y_train_processed, prep_info = lab.prepare_data(
        X_train, y_train, timestamps=ts_train
    )
    X_test_processed, y_test_processed, _ = lab.prepare_data(
        X_test, y_test, timestamps=ts_test
    )
    print(f"   Original features: {X_train.shape[1]}")
    print(f"   Processed features: {X_train_processed.shape[1]}")
    print(f"   Outliers removed: {prep_info['outliers_removed']}")
    
    # Step 5: Train model using LatencyPredictor
    print("\n5. Training latency prediction model...")
    model = LatencyPredictor(
        n_estimators=lab.current_experiment.hyperparameters['n_estimators'],
        max_depth=lab.current_experiment.hyperparameters['max_depth'],
        random_state=lab.current_experiment.hyperparameters['random_state']
    )
    
    import time
    start_time = time.time()
    model.fit(X_train_processed, y_train_processed)
    training_time = time.time() - start_time
    
    training_info = {
        'training_time_seconds': training_time,
        'n_samples': X_train_processed.shape[0],
        'n_features': X_train_processed.shape[1]
    }
    print(f"   Training time: {training_time:.2f}s")
    print(f"   Training R²: {model.metrics['r2_score']:.4f}")
    
    # Step 6: Evaluate model
    print("\n6. Evaluating model...")
    test_metrics = model.evaluate(X_test_processed, y_test_processed)
    
    # Make predictions for inference time measurement
    import time
    start_time = time.time()
    predictions = [model.predict(x.reshape(1, -1)) for x in X_test_processed[:10]]
    inference_time = (time.time() - start_time) / 10 * 1000  # ms per sample
    
    metrics = {
        'accuracy': test_metrics['r2_score'],
        'mae': test_metrics['mae'],
        'rmse': test_metrics['rmse'],
        'mape': test_metrics['mape'],
        'inference_time_ms': inference_time
    }
    
    print(f"   Accuracy (R²): {metrics['accuracy']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f} ms")
    print(f"   RMSE: {metrics['rmse']:.4f} ms")
    print(f"   MAPE: {metrics['mape']:.2f}%")
    print(f"   Inference time: {metrics['inference_time_ms']:.4f} ms/sample")
    
    # Step 7: Save results
    print("\n7. Saving experiment results...")
    result = lab.save_experiment_results(
        model,  # Use LatencyPredictor model
        metrics,
        training_info,
        prep_info
    )
    print(f"   Experiment ID: {result.experiment_id}")
    print(f"   Model ID: {result.model_id}")
    print(f"   Model size: {result.model_size_mb:.2f} MB")
    print(f"   Results saved to: {result.artifacts['config_file']}")
    
    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)
    
    return lab, result


if __name__ == "__main__":
    lab, result = run_latency_experiment()
    
    # Compare experiments (if multiple exist)
    comparison = lab.compare_experiments()
    if not comparison.empty:
        print("\nExperiment Comparison:")
        print(comparison.to_string(index=False))

