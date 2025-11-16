# ML Models for AI Routing Lab

This directory contains ML models and infrastructure copied from CloudBridge AI Service ecosystem.

## Structure

```
models/
├── core/                    # Core ML infrastructure
│   ├── model_registry.py   # Model versioning and management
│   ├── model_validator.py  # Model validation framework
│   ├── data_preprocessor.py # Data preprocessing
│   └── feature_extractor.py # Feature engineering
│
├── routing/                 # Route optimization models
│   ├── route_ensemble.py   # Ensemble route selection
│   ├── random_forest_route.py
│   ├── neural_network_route.py
│   ├── multi_armed_bandit.py
│   └── q_learning_route.py
│
├── prediction/              # Load/latency prediction models
│   ├── load_ensemble.py
│   ├── lstm_forecast.py
│   ├── arima_model.py
│   ├── prophet_model.py
│   └── random_forest_load.py
│
├── anomaly/                 # Anomaly detection models
│   ├── anomaly_ensemble.py
│   ├── isolation_forest.py
│   ├── one_class_svm.py
│   └── lstm_autoencoder.py
│
└── monitoring/             # Model monitoring
    ├── drift_detector.py
    ├── model_monitor.py
    └── retraining_orchestrator.py
```

## Status

**These files are copied from CloudBridge AI Service and need to be adapted for AI Routing Lab.**

### Next Steps

1. **Review and adapt** models for latency/jitter prediction
2. **Update imports** to match AI Routing Lab structure
3. **Modify** for route selection use case
4. **Test** with quic-test integration

## Usage

See `experiments/README.md` for how to use these models in laboratory experiments.

