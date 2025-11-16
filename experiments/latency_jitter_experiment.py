"""
Complete Latency and Jitter Prediction Experiment

Demonstrates full workflow for latency/jitter prediction and route selection.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import logging

from experiments.lab_experiment import LaboratoryExperiment
from models.prediction import LatencyPredictor, JitterPredictor, RoutePredictionEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_route_data(n_samples: int = 1000, n_routes: int = 5):
    """
    Generate sample data for multiple routes.
    
    Args:
        n_samples: Number of samples per route
        n_routes: Number of routes
        
    Returns:
        Tuple of (X_latency, y_latency, X_jitter, y_jitter, route_ids)
    """
    np.random.seed(42)
    
    X_latency_list = []
    y_latency_list = []
    X_jitter_list = []
    y_jitter_list = []
    route_ids = []
    
    for route_id in range(n_routes):
        # Generate features: route characteristics, network conditions
        X_route = np.random.randn(n_samples, 8)
        
        # Route-specific latency (different base latency per route)
        base_latency = 20 + route_id * 5  # Routes have different base latencies
        latency = (
            base_latency +
            5 * X_route[:, 0] +  # Route distance
            3 * X_route[:, 1] +  # Network congestion
            2 * X_route[:, 2] +  # Packet loss impact
            np.random.randn(n_samples) * 2
        )
        latency = np.maximum(latency, 5)
        
        # Jitter depends on latency variability and route stability
        base_jitter = 2 + route_id * 0.5
        jitter = (
            base_jitter +
            0.5 * np.abs(X_route[:, 1]) +  # Congestion variability
            0.3 * np.abs(X_route[:, 2]) +  # Loss variability
            np.random.randn(n_samples) * 0.5
        )
        jitter = np.maximum(jitter, 0.5)
        
        X_latency_list.append(X_route)
        y_latency_list.append(latency)
        X_jitter_list.append(X_route)  # Same features for jitter
        y_jitter_list.append(jitter)
        route_ids.extend([route_id] * n_samples)
    
    # Combine all routes
    X_latency = np.vstack(X_latency_list)
    y_latency = np.hstack(y_latency_list)
    X_jitter = np.vstack(X_jitter_list)
    y_jitter = np.hstack(y_jitter_list)
    route_ids = np.array(route_ids)
    
    return X_latency, y_latency, X_jitter, y_jitter, route_ids


def run_complete_experiment():
    """Run complete latency/jitter prediction and route selection experiment"""
    
    print("=" * 70)
    print("AI Routing Lab - Complete Latency/Jitter Prediction Experiment")
    print("=" * 70)
    
    # Step 1: Generate data
    print("\n1. Generating route data...")
    X_latency, y_latency, X_jitter, y_jitter, route_ids = generate_route_data(
        n_samples=1000, n_routes=5
    )
    print(f"   Generated {len(X_latency)} samples across {len(np.unique(route_ids))} routes")
    
    # Step 2: Split data
    print("\n2. Splitting data...")
    indices = np.arange(len(X_latency))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_latency_train = X_latency[train_idx]
    y_latency_train = y_latency[train_idx]
    X_latency_test = X_latency[test_idx]
    y_latency_test = y_latency[test_idx]
    
    X_jitter_train = X_jitter[train_idx]
    y_jitter_train = y_jitter[train_idx]
    X_jitter_test = X_jitter[test_idx]
    y_jitter_test = y_jitter[test_idx]
    
    print(f"   Training: {len(train_idx)} samples")
    print(f"   Test: {len(test_idx)} samples")
    
    # Step 3: Create experiment
    print("\n3. Creating laboratory experiment...")
    lab = LaboratoryExperiment()
    lab.create_experiment(
        experiment_name="latency_jitter_prediction",
        description="Predict latency and jitter for route selection",
        model_type="route_selection",
        model_framework="sklearn",
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 15,
            'random_state': 42
        },
        data_config={
            'extract_features': True,
            'include_stats': True,
            'include_rolling': False,
            'include_ema': False,
            'remove_outliers': True,
            'normalize': True
        },
        evaluation_metrics=['accuracy', 'mae', 'rmse', 'mape']
    )
    
    # Step 4: Prepare data
    print("\n4. Preparing data...")
    X_latency_train_proc, y_latency_train_proc, prep_info_lat = lab.prepare_data(
        X_latency_train, y_latency_train
    )
    X_jitter_train_proc, y_jitter_train_proc, prep_info_jit = lab.prepare_data(
        X_jitter_train, y_jitter_train
    )
    
    X_latency_test_proc, y_latency_test_proc, _ = lab.prepare_data(
        X_latency_test, y_latency_test
    )
    X_jitter_test_proc, y_jitter_test_proc, _ = lab.prepare_data(
        X_jitter_test, y_jitter_test
    )
    
    print(f"   Latency features: {X_latency_train_proc.shape[1]}")
    print(f"   Jitter features: {X_jitter_train_proc.shape[1]}")
    
    # Step 5: Train models
    print("\n5. Training latency and jitter predictors...")
    
    latency_predictor = LatencyPredictor(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )
    latency_predictor.fit(X_latency_train_proc, y_latency_train_proc)
    
    jitter_predictor = JitterPredictor(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )
    jitter_predictor.fit(X_jitter_train_proc, y_jitter_train_proc)
    
    print(f"   Latency R²: {latency_predictor.metrics['r2_score']:.4f}")
    print(f"   Jitter R²: {jitter_predictor.metrics['r2_score']:.4f}")
    
    # Step 6: Evaluate models
    print("\n6. Evaluating models...")
    latency_metrics = latency_predictor.evaluate(X_latency_test_proc, y_latency_test_proc)
    jitter_metrics = jitter_predictor.evaluate(X_jitter_test_proc, y_jitter_test_proc)
    
    print(f"\n   Latency Prediction:")
    print(f"     R²: {latency_metrics['r2_score']:.4f}")
    print(f"     MAE: {latency_metrics['mae']:.2f} ms")
    print(f"     RMSE: {latency_metrics['rmse']:.2f} ms")
    print(f"     MAPE: {latency_metrics['mape']:.2f}%")
    
    print(f"\n   Jitter Prediction:")
    print(f"     R²: {jitter_metrics['r2_score']:.4f}")
    print(f"     MAE: {jitter_metrics['mae']:.2f} ms")
    print(f"     RMSE: {jitter_metrics['rmse']:.2f} ms")
    print(f"     MAPE: {jitter_metrics['mape']:.2f}%")
    
    # Step 7: Route selection ensemble
    print("\n7. Testing route selection ensemble...")
    ensemble = RoutePredictionEnsemble(
        latency_weight=0.7,
        jitter_weight=0.3,
        latency_predictor=latency_predictor,
        jitter_predictor=jitter_predictor
    )
    ensemble.fitted = True  # Already trained
    
    # Test route selection on a few samples
    test_routes = {}
    for route_id in range(5):
        # Get test samples for this route
        route_mask = route_ids[test_idx] == route_id
        if np.sum(route_mask) > 0:
            sample_idx = np.where(route_mask)[0][0]
            test_routes[route_id] = (
                X_latency_test_proc[sample_idx],
                X_jitter_test_proc[sample_idx]
            )
    
    if test_routes:
        best_route_id, best_prediction = ensemble.select_best_route(test_routes)
        print(f"   Selected route: {best_route_id}")
        print(f"   Predicted latency: {best_prediction.predicted_latency_ms:.2f} ms")
        print(f"   Predicted jitter: {best_prediction.predicted_jitter_ms:.2f} ms")
        print(f"   Combined score: {best_prediction.combined_score:.4f}")
        print(f"   Overall confidence: {best_prediction.overall_confidence:.4f}")
    
    # Step 8: Check accuracy target
    print("\n8. Accuracy target check...")
    latency_accuracy = latency_metrics['r2_score']
    jitter_accuracy = jitter_metrics['r2_score']
    
    print(f"   Latency accuracy: {latency_accuracy:.4f} ({'>92%' if latency_accuracy > 0.92 else '<92%'})")
    print(f"   Jitter accuracy: {jitter_accuracy:.4f} ({'>92%' if jitter_accuracy > 0.92 else '<92%'})")
    
    if latency_accuracy > 0.92 and jitter_accuracy > 0.92:
        print("\n   Target achieved! Both models exceed 92% accuracy!")
    else:
        print("\n   Target not yet achieved. Consider:")
        print("      - More training data")
        print("      - Feature engineering")
        print("      - Hyperparameter tuning")
    
    print("\n" + "=" * 70)
    print("Experiment completed!")
    print("=" * 70)
    
    return {
        'latency_predictor': latency_predictor,
        'jitter_predictor': jitter_predictor,
        'ensemble': ensemble,
        'latency_metrics': latency_metrics,
        'jitter_metrics': jitter_metrics
    }


if __name__ == "__main__":
    results = run_complete_experiment()

