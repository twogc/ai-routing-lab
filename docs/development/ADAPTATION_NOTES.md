# Notes for Adapting CloudBridge AI Service Code

## Files Copied

The following files have been copied from `cloudbridge-ai-service` to `ai-routing-lab`:

### Core Infrastructure
- `models/core/model_registry.py` - Model versioning and management
- `models/core/data_preprocessor.py` - Data preprocessing
- `models/core/feature_extractor.py` - Feature engineering
- `models/core/model_validator.py` - Model validation

### Routing Models - Need Adaptation
- `models/routing/route_ensemble.py` - Ensemble route selection
- `models/routing/random_forest_route.py` - Random Forest route classifier
- `models/routing/neural_network_route.py` - Neural Network optimizer
- `models/routing/multi_armed_bandit.py` - MAB router
- `models/routing/q_learning_route.py` - Q-Learning router

### Prediction Models - Need Adaptation
- `models/prediction/load_ensemble.py` - Load prediction ensemble
- `models/prediction/lstm_forecast.py` - LSTM time series
- `models/prediction/arima_model.py` - ARIMA forecasting
- `models/prediction/prophet_model.py` - Prophet model
- `models/prediction/random_forest_load.py` - Random Forest predictor

### Anomaly Detection - Optional
- `models/anomaly/anomaly_ensemble.py` - Anomaly detection ensemble
- `models/anomaly/isolation_forest.py` - Isolation Forest
- `models/anomaly/one_class_svm.py` - One-Class SVM
- `models/anomaly/lstm_autoencoder.py` - LSTM Autoencoder

### Monitoring - Optional
- `models/monitoring/drift_detector.py` - Model drift detection
- `models/monitoring/model_monitor.py` - Model monitoring
- `models/monitoring/retraining_orchestrator.py` - Retraining orchestration

## Adaptation Checklist

### 1. Update Imports
- [ ] Change `from app.ml.*` to `from models.*`
- [ ] Update relative imports
- [ ] Fix circular dependencies

### 2. Adapt for Latency/Jitter Prediction
- [ ] Modify prediction models for latency prediction
- [ ] Modify prediction models for jitter prediction
- [ ] Update feature extraction for route characteristics
- [ ] Adjust hyperparameters for routing use case

### 3. Route Selection Models
- [ ] Adapt route ensemble for latency/jitter-based selection
- [ ] Update route scoring based on predictions
- [ ] Modify MAB and Q-Learning for route optimization

### 4. Integration with quic-test
- [ ] Update data collectors to use quic-test metrics
- [ ] Map quic-test metrics to model features
- [ ] Integrate validation with quic-test results

### 5. Testing
- [ ] Create unit tests for adapted models
- [ ] Test with quic-test data
- [ ] Validate prediction accuracy >92%

## Priority Order

1. **High Priority:**
   - Core infrastructure (mostly done)
   - Latency prediction models
   - Route selection ensemble

2. **Medium Priority:**
   - Jitter prediction models
   - Feature extraction for routing

3. **Low Priority:**
   - Anomaly detection (if needed)
   - Model monitoring (if needed)

## Notes

- Keep the structure similar to CloudBridge AI Service for consistency
- Focus on latency/jitter prediction accuracy >92%
- Ensure integration with quic-test for validation
- Maintain experiment reproducibility

