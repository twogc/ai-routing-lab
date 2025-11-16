# AI Routing Lab - Architecture Documentation

**Version:** 0.1.0  
**Last Updated:** November 2025

---

## Overview

AI Routing Lab is designed as a modular ML research project that integrates with CloudBridge quic-test for data collection and validation. The architecture follows best practices for ML research projects with clear separation of concerns.

---

## Architecture Layers

### 1. Data Collection Layer

**Purpose:** Collect metrics from quic-test and other sources

**Components:**
- `PrometheusCollector` - Collect metrics from Prometheus endpoint
- `JSONFileCollector` - Watch for JSON files from quic-test
- `DataPipeline` - Process and normalize collected data

**Integration Points:**
- quic-test Prometheus export (port 9090)
- quic-test JSON export files
- CloudBridge Monitoring metrics

### 2. ML Pipeline Layer

**Purpose:** Train and evaluate ML models

**Components:**
- `LatencyPredictor` - Predict route latency
- `JitterPredictor` - Predict route jitter
- `RouteSelector` - Ensemble model for route selection
- `TrainingPipeline` - Automated training workflow

**ML Frameworks:**
- TensorFlow 2.x for deep learning models
- PyTorch 2.x (alternative)
- scikit-learn for traditional ML models

### 3. Inference Layer

**Purpose:** Real-time predictions for route selection

**Components:**
- `Predictor` - Model inference engine
- `RouteOptimizer` - Route selection logic
- `PredictionAPI` - REST/gRPC API for predictions

**Performance Requirements:**
- <10ms inference latency
- 1000+ predictions/second throughput

### 4. Validation Layer

**Purpose:** Validate ML predictions against real QUIC traffic

**Components:**
- `QuicTestValidator` - Integration with quic-test for validation
- `MetricsCalculator` - Calculate prediction accuracy
- `ABTestingFramework` - A/B testing support

**Validation Metrics:**
- Prediction accuracy (>92% target)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### 5. Integration Layer

**Purpose:** Integrate with CloudBridge infrastructure

**Components:**
- `QuicTestClient` - Client for quic-test integration
- `RelayIntegration` - Integration with CloudBridge Relay
- `MetricsBridge` - Bridge between ML and production systems

---

## Data Flow

```
quic-test (Go)
    │
    │ Export metrics (Prometheus/JSON)
    ▼
Data Collection Layer (Python)
    │
    │ Process and normalize
    ▼
ML Pipeline Layer
    │
    │ Train models
    ▼
Model Storage (MLflow)
    │
    │ Load trained models
    ▼
Inference Layer
    │
    │ Generate predictions
    ▼
Validation Layer
    │
    │ Validate with quic-test
    ▼
Integration Layer
    │
    │ Route recommendations
    ▼
CloudBridge Relay (Production)
```

---

## Technology Stack

### Core Technologies
- **Python 3.11+** - Primary language
- **TensorFlow 2.x** - Deep learning framework
- **PyTorch 2.x** - Alternative DL framework
- **scikit-learn** - Traditional ML algorithms

### Data Processing
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scipy** - Scientific computing

### Experiment Tracking
- **MLflow** - Experiment tracking and model registry
- **Weights & Biases** - Optional alternative

### Integration
- **Prometheus Client** - Metrics collection
- **FastAPI** - REST API
- **gRPC** - High-performance RPC

---

## Model Architecture

### Latency Prediction Model

**Input Features:**
- Historical latency (time-series)
- Route characteristics (PoP locations, BGP paths)
- Network conditions (time of day, day of week)
- Current network state (congestion, packet loss)

**Model Architecture:**
- LSTM/GRU for time-series forecasting
- Transformer for sequence modeling (optional)
- Ensemble of multiple models

**Output:**
- Predicted latency (ms)
- Confidence interval

### Jitter Prediction Model

**Input Features:**
- Historical jitter patterns
- Route stability metrics
- Network variability indicators

**Model Architecture:**
- Similar to latency predictor
- Focus on variability prediction

**Output:**
- Predicted jitter (ms)
- Jitter variability estimate

### Route Selection Ensemble

**Input:**
- Latency predictions for all routes
- Jitter predictions for all routes
- Route availability
- Current network state

**Logic:**
- Weighted combination of latency and jitter predictions
- Route ranking based on predicted performance
- Fallback to baseline routing if predictions unavailable

**Output:**
- Ranked list of routes
- Recommended route with confidence score

---

## Integration with quic-test

### Data Collection

**Prometheus Integration:**
```python
from data.collectors.prometheus_collector import PrometheusCollector

collector = PrometheusCollector(prometheus_url="http://localhost:9090")
metrics = collector.collect_all_metrics()
```

**JSON File Integration:**
```python
from data.collectors.json_file_collector import JSONFileCollector

collector = JSONFileCollector(watch_directory="quic-test/results/")
collector.start_watching()
```

### Validation

**Validate Predictions:**
```python
from evaluation.quic_test_validator import QuicTestValidator

validator = QuicTestValidator()
accuracy = validator.validate_predictions(
    predictions_file="predictions.json",
    quic_test_results="quic_test_results.json"
)
```

---

## Performance Considerations

### Model Inference
- **Target Latency:** <10ms per prediction
- **Throughput:** 1000+ predictions/second
- **Model Size:** <100MB per model

### Data Collection
- **Collection Interval:** 15 seconds (aligned with quic-test)
- **Storage:** Time-series database (InfluxDB or similar)
- **Retention:** 30 days for training data

### Training
- **Training Frequency:** Daily retraining on new data
- **Training Time:** <1 hour per model
- **GPU Requirements:** Optional, CPU training sufficient for initial models

---

## Security Considerations

1. **Data Privacy:**
   - No PII in training data
   - Anonymized route identifiers
   - Secure storage of metrics

2. **Model Security:**
   - Signed model artifacts
   - Version control for models
   - Access control for model registry

3. **API Security:**
   - Authentication for prediction API
   - Rate limiting
   - Input validation

---

## Deployment Architecture

### Development Environment
```
Local Machine
├── Python venv
├── quic-test (local)
├── Prometheus (local)
└── MLflow (local)
```

### Production Environment
```
Kubernetes Cluster
├── AI Routing Lab Pods
│   ├── Data Collector
│   ├── ML Training Job
│   ├── Inference Service
│   └── Validation Service
├── Prometheus (existing)
├── MLflow Server
└── CloudBridge Relay (integration)
```

---

## Monitoring and Observability

### Metrics to Track
- Prediction accuracy (real-time)
- Inference latency
- Model drift detection
- Data collection health
- Integration status with quic-test

### Logging
- Structured logging (JSON format)
- Log levels: DEBUG, INFO, WARNING, ERROR
- Integration with CloudBridge monitoring

---

## Future Enhancements

1. **Real-time Learning:**
   - Online learning for model updates
   - Continuous model improvement

2. **Multi-path Optimization:**
   - Simultaneous multi-path routing
   - Load balancing based on predictions

3. **Advanced Features:**
   - Anomaly detection integration
   - DDoS prediction
   - Capacity planning predictions

---

**Last Updated:** November 2025  
**Maintained By:** CloudBridge Research Team

