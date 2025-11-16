# AI Routing Lab - Usage Examples

**Version**: 1.1
**Date**: November 2025

Examples of using improved latency and jitter prediction models with ensemble methods.

---

## Quick Start

### 1. Basic LatencyPredictor Usage

```python
import numpy as np
from models.prediction.latency_predictor import LatencyPredictor
from models.core.feature_extractor import FeatureExtractor

# Create model
latency_model = LatencyPredictor(
    n_estimators=100,
    max_depth=15,
    use_gradient_boosting=False  # Random Forest
)

# Prepare synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 20

# Synthetic routing features
X_train = np.random.randn(n_samples, n_features)
# Synthetic latency values (in milliseconds)
y_train = np.random.randn(n_samples) * 10 + 50

# Feature names
feature_names = [
    "route_hop_count", "bgp_path_length", "pop_distance",
    "historical_latency_p50", "historical_latency_p95",
    "network_congestion", "packet_loss_rate",
    "bandwidth_utilization", "time_of_day",
    "day_of_week", "is_weekend", "upstream_latency",
    "downstream_latency", "routing_policy",
    "failover_count", "rtt_variance", "jitter_p95",
    "interface_load", "buffer_occupancy", "path_redundancy"
]

# Train model with ensemble
latency_model.fit(
    X_train, y_train,
    feature_names=feature_names,
    use_ensemble=True
)

# Print training metrics
metrics = latency_model.get_metrics()
print(f"Training completed!")
print(f"R² Score: {metrics['r2_score']:.4f}")
print(f"MAE: {metrics['mae']:.2f}ms")
print(f"RMSE: {metrics['rmse']:.2f}ms")
print(f"MAPE: {metrics['mape']:.2f}%")

# Make prediction for new route
X_test = np.random.randn(5, n_features)  # 5 routes
predictions = []

for i in range(5):
    X_single = X_test[i:i+1]
    prediction = latency_model.predict(
        X_single,
        return_confidence=True,
        use_ensemble=True
    )
    predictions.append(prediction)

    print(f"\nRoute {i+1}:")
    print(f"  Predicted latency: {prediction.predicted_latency_ms:.2f}ms")
    print(f"  Confidence interval: [{prediction.confidence_interval[0]:.2f}, {prediction.confidence_interval[1]:.2f}]ms")
    print(f"  Confidence score: {prediction.confidence_score:.2%}")
    print(f"  Uncertainty: ±{prediction.uncertainty:.2f}ms")
    print(f"  Prediction time: {prediction.prediction_time_ms:.2f}ms")
```

### 2. Using JitterPredictor

```python
from models.prediction.jitter_predictor import JitterPredictor

# Create jitter model
jitter_model = JitterPredictor(
    n_estimators=100,
    max_depth=15,
    use_gradient_boosting=False
)

# Synthetic jitter data (variability in ms)
y_jitter_train = np.random.randn(n_samples) * 2 + 5

# Train model
jitter_model.fit(
    X_train, y_jitter_train,
    feature_names=feature_names,
    use_ensemble=True
)

# Jitter metrics
jitter_metrics = jitter_model.get_metrics()
print(f"Jitter model: R²={jitter_metrics['r2_score']:.4f}")

# Jitter prediction
jitter_pred = jitter_model.predict(X_test[0:1], use_ensemble=True)
print(f"\nPredicted jitter: {jitter_pred.predicted_jitter_ms:.2f}ms")
print(f"Variability estimate: ±{jitter_pred.variability_estimate:.2f}ms")
print(f"Uncertainty: ±{jitter_pred.uncertainty:.2f}ms")
```

---

## Working with Real Data

### 3. Feature Extraction from Raw Metrics

```python
from models.core.feature_extractor import FeatureExtractor, DomainFeatureExtractor
import pandas as pd

# Load metrics from CSV or database
# For example, from quic-test
metrics_df = pd.read_csv('quic_metrics.csv')

# Prepare raw data
raw_metrics = metrics_df[['latency', 'packet_loss', 'bandwidth', 'cpu_usage']].values

# Extract features
feature_extractor = FeatureExtractor(window_sizes=[5, 10, 20])
feature_result = feature_extractor.extract_all_features(
    X=raw_metrics,
    timestamps=metrics_df['timestamp'].values,
    include_stats=True,
    include_rolling=True,
    include_ema=True
)

print(f"Extracted features: {feature_result.n_features_created}")
print(f"Extraction time: {feature_result.extraction_time_ms:.2f}ms")

# Use for model training
X = feature_result.features
feature_names = feature_result.feature_names

latency_model = LatencyPredictor()
latency_model.fit(X, metrics_df['actual_latency'].values, feature_names=feature_names)
```

### 4. Domain-Specific Features for Routing

```python
domain_extractor = DomainFeatureExtractor()

# Example network metrics
route_metrics = {
    'rtt': 45,                    # Round-trip time in ms
    'packet_loss': 0.05,          # Loss percentage
    'bandwidth_used': 600,        # Used bandwidth in Mbps
    'bandwidth_total': 1000,      # Total bandwidth in Mbps
    'errors': 2,                  # Error count
    'total_requests': 1000,       # Total requests
    'throughput': 800,            # Throughput in Mbps
    'latency_p95': 120,           # P95 latency in ms
    'cpu_usage': 45,              # CPU usage in %
    'memory_usage': 60,           # Memory usage in %
}

# Extract network features
network_features = domain_extractor.extract_network_features(route_metrics)
print("Network Features:", network_features)
# Output: {
#     'rtt_zscore': -0.5,
#     'packet_loss_rate': 0.0005,
#     'bandwidth_utilization': 0.6,
#     'error_rate': 0.002
# }

# Extract performance features
perf_features = domain_extractor.extract_performance_features(route_metrics)
print("Performance Features:", perf_features)
# Output: {
#     'throughput_normalized': 0.8,
#     'latency_p95_normalized': 1.2,
#     'cpu_usage_percent': 0.45,
#     'memory_usage_percent': 0.60
# }
```

---

## Evaluation and Validation

### 5. Model Evaluation on Test Set

```python
# Split data into train/test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
latency_model = LatencyPredictor()
latency_model.fit(X_train, y_train, feature_names=feature_names)

# Evaluate on test set
test_metrics = latency_model.evaluate(X_test, y_test, use_ensemble=True)

print("Test Metrics:")
print(f"  R² Score: {test_metrics['r2_score']:.4f}")
print(f"  MAE: {test_metrics['mae']:.2f}ms")
print(f"  RMSE: {test_metrics['rmse']:.2f}ms")
print(f"  MAPE: {test_metrics['mape']:.2f}%")

# Check accuracy requirements
if test_metrics['r2_score'] > 0.92:
    print("Accuracy requirement met (R² > 0.92)")
else:
    print("Accuracy requirement NOT met")
```

### 6. Feature Importance Analysis

```python
# Get feature importance
feature_importance = latency_model.get_feature_importance()

# Print top-10 features
sorted_features = sorted(
    feature_importance.items(),
    key=lambda x: x[1],
    reverse=True
)

print("Top-10 important features for latency prediction:")
for i, (feature, importance) in enumerate(sorted_features[:10], 1):
    print(f"{i:2d}. {feature:30s} {importance:.4f}")
```

---

## Batch Prediction

### 7. Prediction for Multiple Routes

```python
# Predict latency and jitter for multiple routes
def predict_route_quality(X_routes, latency_model, jitter_model):
    """
    Predict quality of multiple routes

    Args:
        X_routes: Features matrix (n_routes, n_features)
        latency_model: Trained LatencyPredictor
        jitter_model: Trained JitterPredictor

    Returns:
        List of route quality predictions
    """
    results = []

    for i, X_single in enumerate(X_routes):
        latency_pred = latency_model.predict(
            X_single.reshape(1, -1),
            use_ensemble=True
        )
        jitter_pred = jitter_model.predict(
            X_single.reshape(1, -1),
            use_ensemble=True
        )

        # Combined route score (70% latency, 30% jitter)
        route_quality_score = (
            0.7 * (1 - latency_pred.confidence_score) +
            0.3 * (1 - jitter_pred.confidence_score)
        )

        results.append({
            'route_id': i,
            'predicted_latency_ms': latency_pred.predicted_latency_ms,
            'latency_ci': latency_pred.confidence_interval,
            'predicted_jitter_ms': jitter_pred.predicted_jitter_ms,
            'jitter_ci': jitter_pred.confidence_interval,
            'quality_score': route_quality_score,
            'rank': 0  # Will be set later
        })

    # Rank routes by quality
    results.sort(key=lambda x: x['quality_score'])
    for i, result in enumerate(results, 1):
        result['rank'] = i

    return results

# Usage
X_routes = np.random.randn(10, n_features)  # 10 routes
route_predictions = predict_route_quality(X_routes, latency_model, jitter_model)

# Print results
for pred in route_predictions:
    print(f"Route {pred['route_id']} (Rank {pred['rank']}):")
    print(f"  Latency: {pred['predicted_latency_ms']:.2f}ms "
          f"[{pred['latency_ci'][0]:.2f}, {pred['latency_ci'][1]:.2f}]")
    print(f"  Jitter: {pred['predicted_jitter_ms']:.2f}ms "
          f"[{pred['jitter_ci'][0]:.2f}, {pred['jitter_ci'][1]:.2f}]")
    print(f"  Quality Score: {pred['quality_score']:.4f}")
```

---

## Single vs Ensemble Comparison

### 8. Benchmark: Single Model vs Ensemble

```python
import time

# Model without ensemble
single_model = LatencyPredictor(n_estimators=100)
single_model.fit(X_train, y_train, use_ensemble=False)

# Model with ensemble
ensemble_model = LatencyPredictor(n_estimators=100)
ensemble_model.fit(X_train, y_train, use_ensemble=True)

# Benchmarking
X_benchmark = X_test[:100]  # 100 samples
n_iterations = 100

# Single model
start = time.time()
for _ in range(n_iterations):
    single_model.predict(X_benchmark, use_ensemble=False)
single_time = time.time() - start

# Ensemble model
start = time.time()
for _ in range(n_iterations):
    ensemble_model.predict(X_benchmark, use_ensemble=True)
ensemble_time = time.time() - start

# Evaluation comparison
single_metrics = single_model.evaluate(X_test, y_test, use_ensemble=False)
ensemble_metrics = ensemble_model.evaluate(X_test, y_test, use_ensemble=True)

print("PERFORMANCE COMPARISON:")
print("-" * 60)
print(f"{'Metric':<20} {'Single Model':<20} {'Ensemble':<20}")
print("-" * 60)
print(f"{'R² Score':<20} {single_metrics['r2_score']:<20.4f} {ensemble_metrics['r2_score']:<20.4f}")
print(f"{'MAE (ms)':<20} {single_metrics['mae']:<20.2f} {ensemble_metrics['mae']:<20.2f}")
print(f"{'RMSE (ms)':<20} {single_metrics['rmse']:<20.2f} {ensemble_metrics['rmse']:<20.2f}")
print(f"{'MAPE (%)':<20} {single_metrics['mape']:<20.2f} {ensemble_metrics['mape']:<20.2f}")
print(f"{'Pred Time (s)':<20} {single_time:<20.2f} {ensemble_time:<20.2f}")
print("-" * 60)
```

---

## Integration with Experiment Framework

### 9. Using in Experiments

```python
from experiments.lab_experiment import LabExperiment
from experiments.latency_jitter_experiment import LatencyJitterExperiment

# Create experiment
experiment = LatencyJitterExperiment(
    name="ensemble_latency_jitter_v1",
    description="Test ensemble models for latency and jitter prediction"
)

# Run experiment
results = experiment.run(
    X_train=X_train,
    y_latency_train=y_train,
    y_jitter_train=y_jitter_train,
    X_test=X_test,
    y_latency_test=y_test,
    y_jitter_test=y_jitter_test,
    use_ensemble=True
)

# Results
print("\nExperiment Results:")
print(f"Latency R²: {results['latency_r2']:.4f}")
print(f"Jitter R²: {results['jitter_r2']:.4f}")
print(f"Total time: {results['total_time']:.2f}s")
```

---

## Error Handling

### 10. Error Handling

```python
from models.prediction.latency_predictor import LatencyPredictor

latency_model = LatencyPredictor()

# Error 1: Predict before fit
try:
    X_test = np.random.randn(1, 20)
    latency_model.predict(X_test)
except RuntimeError as e:
    print(f"Error: {e}")
    # Output: "Model must be fitted before prediction"

# Error 2: Wrong feature count
try:
    latency_model.fit(X_train, y_train, feature_names=feature_names)
    X_wrong = np.random.randn(1, 10)  # Wrong number of features
    latency_model.predict(X_wrong)
except ValueError as e:
    print(f"Error: {e}")
    # Output: "Expected 20 features, got 10"

# Correct usage
latency_model.fit(X_train, y_train, feature_names=feature_names)
X_correct = np.random.randn(1, 20)
prediction = latency_model.predict(X_correct)
print(f"Prediction completed: {prediction.predicted_latency_ms:.2f}ms")
```

---

## Performance Optimization

### 11. Performance Optimization

```python
# Parameters for fast prediction
fast_model = LatencyPredictor(
    n_estimators=50,      # Fewer trees
    max_depth=10,         # Less depth
    min_samples_split=10, # More nodes
    min_samples_leaf=5    # More leaves
)

fast_model.fit(X_train, y_train, use_ensemble=True)

# Fast batch prediction
X_batch = X_test[:1000]
start = time.time()
for X_single in X_batch:
    pred = fast_model.predict(X_single.reshape(1, -1), use_ensemble=False)
duration = time.time() - start

print(f"1000 predictions in {duration:.2f}s ({1000/duration:.0f} pred/sec)")
```

---

## Model Saving and Loading

### 12. Model Persistence

```python
import pickle

# Save model
model_path = 'models/saved/latency_model_v1.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(latency_model, f)

print(f"Model saved to {model_path}")

# Load model
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

# Use loaded model
prediction = loaded_model.predict(X_test[0:1])
print(f"Loaded model works: {prediction.predicted_latency_ms:.2f}ms")
```

---

## Conclusion

These examples demonstrate the main capabilities of the new models:
- Training with ensemble methods
- Prediction with confidence intervals
- Feature extraction and engineering
- Batch processing of routes
- Model evaluation and optimization

For more complex scenarios, see `TECHNICAL_SPECIFICATION.md` and the model source code.
