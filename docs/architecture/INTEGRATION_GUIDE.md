# AI Routing Lab - Integration Guide

**How to integrate AI Routing Lab with quic-test and CloudBridge Relay**

---

## Integration Overview

AI Routing Lab integrates with two main components:

1. **quic-test** - For data collection and model validation
2. **CloudBridge Relay** - For production route optimization

---

## Integration with quic-test

### Step 1: Setup quic-test with Prometheus Export

```bash
# Start quic-test server with Prometheus metrics
cd cloudbridge/quic-test
./bin/quic-server --prometheus-port 9090 --prometheus-path /metrics
```

### Step 2: Configure Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'quic-test'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9090']
```

### Step 3: Collect Data

```python
from data.collectors.quic_test_collector import PrometheusCollector

collector = PrometheusCollector(prometheus_url="http://localhost:9090")
metrics = collector.collect_all_metrics()
```

### Step 4: Validate Predictions

```bash
# Run quic-test with ML validation mode
./bin/quic-client \
  --ml-validation-mode \
  --ml-predictions-file predictions.json \
  --output results.json
```

---

## Integration with CloudBridge Relay

### Step 1: Deploy ML Prediction API

```bash
# Start inference service
python inference/predictor.py --model models/latency_predictor.pkl --port 8080
```

### Step 2: Configure CloudBridge Relay

```yaml
# relay-config.yaml
ml_routing:
  enabled: true
  prediction_api_url: "http://ai-routing-lab:8080/api/v1/predict"
  fallback_to_baseline: true
  prediction_timeout: 100ms
```

### Step 3: Route Selection

CloudBridge Relay will query ML API for route predictions and select optimal route based on predictions.

---

## Data Collection Workflow

1. **quic-test generates traffic** → Exports metrics to Prometheus
2. **AI Routing Lab collects metrics** → Processes and stores
3. **ML models train** → Generate predictions
4. **Predictions validated** → Against quic-test results
5. **Production deployment** → CloudBridge Relay uses predictions

---

## Validation Workflow

1. **Generate predictions** for test routes
2. **Run quic-test** with same routes
3. **Compare predictions vs actual** latency/jitter
4. **Calculate accuracy** metrics
5. **Report results** and iterate

---

**For detailed integration code, see:** `integration/quic_test_client.py` and `integration/relay_integration.py`

