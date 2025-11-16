# AI Routing Lab

**Predictive Route Selection using Machine Learning for Latency/Jitter Optimization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Research-green)]()

**Available Documentation:**
- **English:** This document (README.md)
- **Russian:** [README_ru.md](README_ru.md) - Русская документация проекта

---

## Project Overview

AI Routing Lab is a research project focused on developing machine learning models for predictive route selection in CloudBridge network infrastructure. The project aims to achieve **>92% accuracy** in predicting latency and jitter for optimal route selection.

**Key Objectives:**
- Predictive route selection based on latency/jitter prediction
- Integration with [quic-test](https://github.com/twogc/quic-test) for model validation on real QUIC traffic
- Production integration with CloudBridge Relay for real-time routing optimization

---

## About CloudBridge Research

### Autonomous Non-Commercial Organization

This project is part of **CloudBridge Research Center** — an independent research center specializing in network technologies, distributed systems, and cybersecurity.

**Organization:**
- Conducts fundamental and applied research in network protocols (QUIC, MASQUE, BGP, etc.)
- Develops and distributes open-source software
- Provides educational programs and training
- Collaborates with leading universities and research institutions
- Prepares highly qualified specialists for industry

**Contact & Resources:**
- **Website:** https://cloudbridge-research.ru/
- **GitHub:** [CloudBridge Research](https://github.com/twogc/cloudbridge-research)
- **Email:** info@cloudbridge-research.ru

---

## Research Goals

### Primary Goal
Develop ML models that can predict route latency and jitter with **>92% accuracy** to enable proactive route selection in CloudBridge network.

### Research Areas

1. **Latency Prediction**
   - Time-series forecasting of route latency
   - Multi-path latency comparison
   - Historical pattern analysis

2. **Jitter Prediction**
   - Jitter variability modeling
   - Network condition impact analysis
   - Route stability assessment

3. **Route Selection Optimization**
   - Ensemble models for route ranking
   - Real-time prediction inference
   - Integration with CloudBridge Relay

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              AI Routing Lab (Python)                    │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Data Collection                                 │   │
│  │  • Prometheus metrics from quic-test             │   │
│  │  • JSON export from quic-test                    │   │
│  │  • Historical data storage                       │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                              │
│                          ▼                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  ML Pipeline                                     │   │
│  │  • LatencyPredictor (Random Forest)              │   │
│  │  • JitterPredictor (Random Forest)               │   │
│  │  • RoutePredictionEnsemble                       │   │
│  │  • Feature engineering                           │   │
│  │  • Model evaluation                              │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                              │
│                          ▼                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Inference Engine                                │   │
│  │  • Real-time predictions                         │   │
│  │  • Route optimization                            │   │
│  │  • API for CloudBridge Relay                     │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                              │
│                          │ Validation                   │
│                          ▼                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  quic-test (Go)                                  │   │
│  │  • Real QUIC traffic generation                  │   │
│  │  • Metrics collection                            │   │
│  │  • ML prediction validation                      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ai-routing-lab/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
│
├── data/                        # Data collection and processing
│   ├── collectors/
│   │   └── quic_test_collector.py    # quic-test integration
│   └── pipelines/
│
├── models/                      # ML model definitions
│   ├── core/                    # Core ML infrastructure
│   │   ├── model_registry.py    # Model versioning
│   │   ├── data_preprocessor.py  # Data preprocessing
│   │   └── feature_extractor.py # Feature engineering
│   ├── prediction/              # Prediction models
│   │   ├── latency_predictor.py # Latency prediction
│   │   ├── jitter_predictor.py  # Jitter prediction
│   │   └── route_prediction_ensemble.py # Route selection
│   ├── routing/                 # Route optimization models
│   ├── anomaly/                 # Anomaly detection (optional)
│   └── monitoring/              # Model monitoring (optional)
│
├── training/                    # Training scripts
│
├── inference/                   # Inference engine
│
├── evaluation/                  # Model evaluation
│
├── experiments/                 # Laboratory experiments
│   ├── lab_experiment.py       # Experiment framework
│   ├── example_experiment.py   # Example experiment
│   └── latency_jitter_experiment.py # Complete workflow
│
└── docs/                        # Documentation
    ├── ARCHITECTURE.md          # Architecture documentation
    └── INTEGRATION_GUIDE.md     # Integration guide
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [quic-test](https://github.com/twogc/quic-test) running and exporting metrics
- Prometheus (optional, for metrics collection)

### Installation

```bash
# Clone the repository
git clone https://github.com/twogc/ai-routing-lab.git
cd ai-routing-lab

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```bash
# Run example latency prediction experiment
python experiments/example_experiment.py

# Run complete latency/jitter experiment
python experiments/latency_jitter_experiment.py

# Collect data from quic-test
python -m data.collectors.quic_test_collector --prometheus-url http://localhost:9090
```

---

## Models

### LatencyPredictor
Random Forest model for predicting route latency.

**Features:**
- Historical latency patterns
- Route characteristics (PoP locations, BGP paths)
- Network conditions (congestion, packet loss)
- Time-based features

**Target:** >92% accuracy (R² score)

### JitterPredictor
Random Forest model for predicting route jitter variability.

**Features:**
- Historical jitter patterns
- Route stability metrics
- Network variability indicators

**Target:** >92% accuracy (R² score)

### RoutePredictionEnsemble
Combines latency and jitter predictions for optimal route selection.

**Scoring:**
- Latency weight: 70%
- Jitter weight: 30%
- Selects route with best combined score

**Target:** >95% optimal route selection

---

## Integration with quic-test

AI Routing Lab integrates with [quic-test](https://github.com/twogc/quic-test) for:

1. **Data Collection:**
   - Prometheus metrics export from quic-test
   - JSON export for historical data
   - Real-time metrics streaming

2. **Model Validation:**
   - Validate ML predictions against real QUIC traffic
   - Compare predicted vs actual latency/jitter
   - Calculate prediction accuracy metrics

3. **Production Testing:**
   - Test route selection in controlled environment
   - A/B testing framework
   - Performance benchmarking

### Setup Integration

1. **Start quic-test with Prometheus export:**
   ```bash
   cd cloudbridge/quic-test
   ./bin/quic-server --prometheus-port 9090
   ```

2. **Collect metrics:**
   ```python
   from data.collectors.quic_test_collector import PrometheusCollector
   
   collector = PrometheusCollector(prometheus_url="http://localhost:9090")
   metrics = collector.collect_all_metrics()
   ```

---

## Laboratory Experiments

The project includes a comprehensive laboratory experiment framework:

```python
from experiments.lab_experiment import create_latency_prediction_experiment
from models.prediction import LatencyPredictor

# Create experiment
lab = create_latency_prediction_experiment()

# Prepare data
X_train_proc, y_train_proc, _ = lab.prepare_data(X_train, y_train)

# Train model
model = LatencyPredictor(n_estimators=100, max_depth=15)
model.fit(X_train_proc, y_train_proc)

# Evaluate
metrics = model.evaluate(X_test_proc, y_test_proc)
print(f"Accuracy (R²): {metrics['r2_score']:.4f}")
```

See `experiments/README.md` for detailed documentation.

---

## Documentation

### Architecture
- [Architecture Documentation](docs/architecture/ARCHITECTURE.md)
- [Integration Guide](docs/architecture/INTEGRATION_GUIDE.md)

### Guides
- [Usage Examples](docs/guides/USAGE_EXAMPLES.md)
- [Contributing Guide](docs/guides/CONTRIBUTING.md)

### Development
- [Adaptation Status](docs/development/ADAPTATION_STATUS.md)
- [Adaptation Notes](docs/development/ADAPTATION_NOTES.md)
- [Upgrade Summary](docs/development/UPGRADE_SUMMARY.md)

### Experiments
- [Laboratory Experiments](experiments/README.md)

### Reports
- [Laboratory Reports](reports/README.md) - Test reports organized by date and version

---

## Technology Stack

- **Language:** Python 3.11+
- **ML Framework:** scikit-learn (Random Forest), TensorFlow/PyTorch (optional)
- **Experiment Tracking:** MLflow
- **Data Processing:** pandas, numpy
- **Metrics Collection:** prometheus-client
- **API:** FastAPI / gRPC

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Related Projects

- [quic-test](https://github.com/twogc/quic-test) - QUIC protocol testing tool
- [CloudBridge Relay](https://github.com/twogc/cloudbridge-scalable-relay) - Production relay server
- [CloudBridge Research](https://github.com/twogc/cloudbridge-research) - Research center

---

## Contact

- **GitHub:** [@twogc](https://github.com/twogc)
- **Email:** info@cloudbridge-research.ru
- **Website:** [cloudbridge-research.ru](https://cloudbridge-research.ru/)

---

## Acknowledgments

This research project is part of [CloudBridge Research Center](https://github.com/twogc/cloudbridge-research) and integrates with the [quic-test](https://github.com/twogc/quic-test) testing framework.

Models and infrastructure adapted from [CloudBridge AI Service](https://github.com/twogc/cloudbridge-ai-service) ecosystem.

---

**Status:** In Active Development  
**Last Updated:** November 2025
