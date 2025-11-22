# AI Routing Lab

**Predictive Route Selection using Machine Learning for Latency/Jitter Optimization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Research-green)]()

**Available Documentation:**
- **English:** This document (README_en.md)
- **Russian:** [README.md](README.md) - Russian documentation

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
├── README.md                    # Main documentation (Russian)
├── README_en.md                 # Documentation (English)
├── QUICKSTART.md                # Quick start guide
├── DOCKER.md                    # Docker deployment guide
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── requirements-dev.txt         # Development dependencies
├── pyproject.toml               # Project configuration
├── pytest.ini                   # Pytest configuration
├── Makefile                     # Automation commands
├── Dockerfile                   # Docker image
├── docker-compose.yml           # Docker Compose configuration
├── setup.py                     # Package setup
│
├── data/                        # Data collection and processing
│   ├── collectors/
│   │   └── quic_test_collector.py    # quic-test integration
│   └── pipelines/
│       └── data_pipeline.py     # Data processing pipeline
│
├── models/                      # ML model definitions
│   ├── core/                    # Core ML infrastructure
│   │   ├── model_registry.py    # Model versioning
│   │   ├── data_preprocessor.py  # Data preprocessing
│   │   ├── feature_extractor.py # Feature engineering
│   │   └── model_validator.py   # Model validation
│   ├── prediction/              # Prediction models
│   │   ├── latency_predictor.py # Latency prediction
│   │   ├── jitter_predictor.py  # Jitter prediction
│   │   └── route_prediction_ensemble.py # Route selection
│   ├── routing/                 # Route optimization models
│   ├── anomaly/                 # Anomaly detection
│   └── monitoring/              # Model monitoring
│
├── training/                    # Training scripts
│   ├── train_latency_model.py   # Train latency model
│   └── train_jitter_model.py    # Train jitter model
│
├── inference/                   # Inference engine
│   └── predictor_service.py     # FastAPI service
│
├── evaluation/                  # Model evaluation
│   └── model_evaluator.py       # Evaluation and validation
│
├── experiments/                 # Laboratory experiments
│   ├── lab_experiment.py       # Experiment framework
│   ├── example_experiment.py   # Example experiment
│   └── latency_jitter_experiment.py # Complete workflow
│
├── tests/                       # Tests
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── e2e/                     # End-to-end tests
│
├── monitoring/                  # Monitoring
│   └── prometheus.yml           # Prometheus configuration
│
└── docs/                        # Documentation
    ├── architecture/            # Architecture
    ├── development/             # Development
    └── guides/                  # Guides
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [quic-test](https://github.com/twogc/quic-test) running and exporting metrics
- Prometheus (optional, for metrics collection)

### Installation

#### Option 1: Using Makefile (Recommended)

```bash
# Clone the repository
git clone https://github.com/twogc/ai-routing-lab.git
cd ai-routing-lab

# Install all development dependencies
make install-dev
```

#### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/twogc/ai-routing-lab.git
cd ai-routing-lab

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Basic Usage

#### Training Models

```bash
# Train latency model
python training/train_latency_model.py \
  --data-path data/training_data.json \
  --model-output models/ \
  --n-estimators 100

# Train jitter model
python training/train_jitter_model.py \
  --data-path data/training_data.json \
  --model-output models/
```

#### Running Inference Service

```bash
# Start FastAPI service
python -m uvicorn inference.predictor_service:app --host 0.0.0.0 --port 5000

# Or using Docker
make docker-up
```

#### Running Experiments

```bash
# Run example latency prediction experiment
python experiments/example_experiment.py

# Run complete latency/jitter experiment
python experiments/latency_jitter_experiment.py
```

#### Collecting Data from quic-test

```bash
# Collect metrics from Prometheus
python -m data.collectors.quic_test_collector --prometheus-url http://localhost:9090
```

### Testing

```bash
# Run all tests
make test

# Unit tests only
make test-unit

# With coverage report
pytest --cov=. --cov-report=html
```

### Docker

```bash
# Build image
make docker-build

# Start all services (API, Prometheus, Grafana, MLflow)
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

For more details, see [DOCKER.md](DOCKER.md) and [QUICKSTART.md](QUICKSTART.md).

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

## Usage Examples

### Predicting Latency for a Single Route

```python
from models.prediction import LatencyPredictor
from models.core.model_registry import ModelRegistry
import numpy as np

# Create and train model
model = LatencyPredictor(n_estimators=100, max_depth=15)

# Prepare data
X_train = np.array([
    [25.5, 2.3, 0.95, 1.0],
    [30.1, 3.1, 0.85, 1.2],
    # ... more data
])
y_train = np.array([26.0, 31.0, ...])  # Actual latency values

# Train
model.fit(X_train, y_train)

# Prepare features for prediction
features = np.array([[25.5, 2.3, 0.95, 1.0]])

# Predict
prediction = model.predict(features)
print(f"Predicted latency: {prediction.predicted_latency_ms:.2f} ms")
print(f"Confidence interval: {prediction.confidence_interval}")
print(f"Confidence: {prediction.confidence_score:.2%}")

# Save model via ModelRegistry
registry = ModelRegistry(models_dir="models/")
registry.register_model(
    model_id="latency_predictor",
    model=model,
    model_type="prediction",
    accuracy=0.95,
    framework="scikit-learn"
)
```

### Selecting Optimal Route from Multiple Routes

```python
from models.prediction import LatencyPredictor, JitterPredictor, RoutePredictionEnsemble
import numpy as np

# Create and train models
latency_model = LatencyPredictor()
jitter_model = JitterPredictor()

# Train on data
X_train = np.random.randn(100, 4)
y_latency = np.random.randn(100) * 10 + 25
y_jitter = np.random.randn(100) * 2 + 2

latency_model.fit(X_train, y_latency)
jitter_model.fit(X_train, y_jitter)

# Create ensemble
ensemble = RoutePredictionEnsemble(
    latency_model=latency_model,
    jitter_model=jitter_model
)

# Data for multiple routes (combined features)
routes_features = {
    'route_0': np.array([[25.5, 2.3, 0.95, 1.0]]),
    'route_1': np.array([[30.1, 3.1, 0.85, 1.2]]),
    'route_2': np.array([[20.3, 1.8, 0.98, 0.8]]),
}

# Predict and select optimal route
best_route, predictions = ensemble.select_best_route(routes_features)
print(f"Optimal route: {best_route}")
```

### Using REST API

```python
import requests

# Predict for a single route
response = requests.post('http://localhost:5000/predict', json={
    'features': [25.5, 2.3, 0.95, 1.0],
    'route_id': 'route_0'
})
print(response.json())

# Compare multiple routes
response = requests.post('http://localhost:5000/predict/routes', json={
    'routes': {
        'route_0': [25.5, 2.3, 0.95, 1.0],
        'route_1': [30.1, 3.1, 0.85, 1.2],
        'route_2': [20.3, 1.8, 0.98, 0.8]
    }
})
result = response.json()
print(f"Best route: {result['best_route']}")
print(f"Ranking: {result['ranking']}")
```

---

## Makefile Commands

The project includes a Makefile with convenient commands for development:

```bash
# Show all available commands
make help

# Installation
make install          # Install dependencies
make install-dev      # Install dev dependencies + pre-commit

# Testing
make test             # Run all tests with coverage
make test-unit        # Unit tests only
make test-integration # Integration tests only

# Code Quality
make lint             # Check code with linters
make format           # Auto-format code
make check            # Full check (lint + test + security)

# Docker
make docker-build     # Build Docker image
make docker-up        # Start all services
make docker-down      # Stop containers
make docker-logs      # View logs
make docker-shell     # Connect to container

# Training Models
make train-latency    # Train latency model
make train-jitter     # Train jitter model

# Utilities
make clean            # Clean temporary files
make security         # Security check
```

For more details, see [QUICKSTART.md](QUICKSTART.md).

---

## Documentation

### Quick Start
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide after updates
- [DOCKER.md](DOCKER.md) - Docker deployment guide

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

---

## Contact

- **GitHub:** [@twogc](https://github.com/twogc)
- **Email:** info@cloudbridge-research.ru
- **Website:** [cloudbridge-research.ru](https://cloudbridge-research.ru/)

---

## Acknowledgments

Models and infrastructure adapted from 2GC CloudBridge Global Network ecosystem.

---

## Development Infrastructure

### Testing

The project includes comprehensive testing infrastructure:
- **62 unit tests** for core components
- **Integration tests** for quic-test integration
- **E2E tests** for complete workflow
- **Coverage:** 22.73% (target: 70%+)

### CI/CD

Automated pipeline includes:
- Automatic test runs on Python 3.11 and 3.12
- Code quality checks (black, isort, flake8, mypy, ruff)
- Security scanning (bandit, safety)
- Coverage reporting

### Development Tools

- **Makefile** - convenient commands for all operations
- **Pre-commit hooks** - automatic checks before commit
- **Docker** - full containerization with Prometheus, Grafana, MLflow
- **Linters and formatters** - maintaining code quality

For more details, see [QUICKSTART.md](QUICKSTART.md) and [Makefile](Makefile).

---

**Status:** In Active Development  
**Last Updated:** November 2025  
**Version:** 0.2.1
