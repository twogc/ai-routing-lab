# Laboratory Experiments Framework

This directory contains the framework for creating and running ML laboratory experiments for AI Routing Lab.

## Overview

The Laboratory Experiment framework provides:

- **Experiment Configuration**: Define experiment parameters, hyperparameters, and evaluation metrics
- **Data Preparation**: Automated feature extraction and preprocessing
- **Model Training**: Standardized training pipeline
- **Model Evaluation**: Comprehensive evaluation metrics
- **Results Tracking**: Save and compare experiment results
- **Model Registry**: Version control and management of trained models

## Components

### Core Infrastructure

The framework uses components from `models/core/`:

- **ModelRegistry**: Manages model versioning, caching, and lifecycle
- **DataPreprocessor**: Handles data cleaning, normalization, and transformation
- **FeatureExtractor**: Generates domain-specific features from raw data

### Laboratory Experiment Framework

**`lab_experiment.py`** provides:

- `LaboratoryExperiment`: Main experiment framework class
- `ExperimentConfig`: Configuration for experiments
- `ExperimentResult`: Results from experiments
- Helper functions for creating standard experiments

## Quick Start

### Basic Example

```python
from experiments.lab_experiment import create_latency_prediction_experiment
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Create experiment
lab = create_latency_prediction_experiment()

# Prepare your data
X_train, y_train = ...  # Your training data
X_test, y_test = ...    # Your test data

# Prepare data (feature extraction + preprocessing)
X_train_processed, y_train_processed, _ = lab.prepare_data(X_train, y_train)
X_test_processed, y_test_processed, _ = lab.prepare_data(X_test, y_test)

# Train model
model = RandomForestRegressor(n_estimators=100)
trained_model, training_info = lab.train_model(
    X_train_processed, y_train_processed, model
)

# Evaluate model
metrics = lab.evaluate_model(trained_model, X_test_processed, y_test_processed)

# Save results
result = lab.save_experiment_results(
    trained_model, metrics, training_info, {}
)
```

### Running Example Experiment

```bash
# Run example latency prediction experiment
python experiments/example_experiment.py
```

## Experiment Types

### Latency Prediction

Predict route latency based on network conditions and route characteristics.

```python
lab = create_latency_prediction_experiment(
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 10
    }
)
```

### Jitter Prediction

Predict route jitter variability.

```python
lab = create_jitter_prediction_experiment(
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 10
    }
)
```

### Custom Experiment

Create a custom experiment with full control:

```python
lab = LaboratoryExperiment()

lab.create_experiment(
    experiment_name="custom_experiment",
    description="Custom experiment description",
    model_type="latency",
    model_framework="tensorflow",
    hyperparameters={
        'learning_rate': 0.001,
        'epochs': 100
    },
    data_config={
        'extract_features': True,
        'remove_outliers': True,
        'normalize': True
    },
    evaluation_metrics=['accuracy', 'mae', 'rmse']
)
```

## Evaluation Metrics

Supported evaluation metrics:

- **accuracy**: R² score for regression, accuracy for classification
- **mae**: Mean Absolute Error
- **rmse**: Root Mean Squared Error
- **mape**: Mean Absolute Percentage Error

## Experiment Results

Results are saved in `experiments/results/`:

- `{experiment_id}_config.json`: Experiment configuration
- `{experiment_id}_results.json`: Experiment results
- Models are registered in Model Registry

## Comparing Experiments

Compare multiple experiments:

```python
comparison = lab.compare_experiments()
print(comparison)
```

## Integration with quic-test

The framework integrates with quic-test for data collection:

```python
from data.collectors.quic_test_collector import PrometheusCollector

# Collect data from quic-test
collector = PrometheusCollector(prometheus_url="http://localhost:9090")
metrics = collector.collect_all_metrics()

# Use metrics for experiment
X, y = prepare_features_from_metrics(metrics)
```

## Best Practices

1. **Reproducibility**: Always set random seeds in hyperparameters
2. **Data Validation**: Validate data quality before experiments
3. **Feature Engineering**: Use FeatureExtractor for domain-specific features
4. **Model Versioning**: Models are automatically versioned in Model Registry
5. **Results Tracking**: Always save experiment results for comparison

## Source

This framework is based on the CloudBridge AI Service ecosystem patterns, adapted for AI Routing Lab research needs.

