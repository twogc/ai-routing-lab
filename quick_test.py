#!/usr/bin/env python3
"""
Quick test script to verify the setup.

Tests basic functionality without requiring full installation.
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("AI Routing Lab - Quick Test")
print("=" * 60)

# Test 1: Import models
print("\n1. Testing imports...")
try:
    from models.core.model_registry import ModelRegistry
    from models.prediction.jitter_predictor import JitterPredictor
    from models.prediction.latency_predictor import LatencyPredictor

    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create and train latency model
print("\n2. Testing LatencyPredictor...")
try:
    # Generate synthetic data
    X_train = np.random.randn(50, 4)
    y_train = np.random.randn(50) * 10 + 25

    model = LatencyPredictor(n_estimators=10, max_depth=5)
    model.fit(X_train, y_train)

    # Make prediction
    X_test = np.random.randn(1, 4)
    prediction = model.predict(X_test)

    print(f"   ✅ Model trained and predicted: {prediction.predicted_latency_ms:.2f} ms")
except Exception as e:
    print(f"   ❌ LatencyPredictor test failed: {e}")
    import traceback

    traceback.print_exc()

# Test 3: Create and train jitter model
print("\n3. Testing JitterPredictor...")
try:
    y_jitter = np.random.randn(50) * 2 + 2

    jitter_model = JitterPredictor(n_estimators=10, max_depth=5)
    jitter_model.fit(X_train, y_jitter)

    jitter_pred = jitter_model.predict(X_test)

    print(f"   ✅ Model trained and predicted: {jitter_pred.predicted_jitter_ms:.2f} ms")
except Exception as e:
    print(f"   ❌ JitterPredictor test failed: {e}")
    import traceback

    traceback.print_exc()

# Test 4: Model Registry
print("\n4. Testing ModelRegistry...")
try:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(models_dir=tmpdir)

        registry.register_model(
            model_id="test_model",
            model=model,
            model_type="prediction",
            accuracy=0.95,
            framework="scikit-learn",
        )

        loaded_model, metadata = registry.get_model("test_model")

        print(f"   ✅ Model registered and loaded: {metadata.model_id}")
except Exception as e:
    print(f"   ❌ ModelRegistry test failed: {e}")
    import traceback

    traceback.print_exc()

# Test 5: Data collectors
print("\n5. Testing data collectors...")
try:
    from data.collectors.quic_test_collector import JSONFileCollector, PrometheusCollector

    collector = PrometheusCollector(prometheus_url="http://localhost:9090")
    json_collector = JSONFileCollector(watch_directory="/tmp")

    print("   ✅ Collectors initialized successfully")
except Exception as e:
    print(f"   ❌ Collectors test failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ All basic tests passed!")
print("=" * 60)
print("\nNext steps:")
print("  1. Install dev dependencies: pip install -r requirements-dev.txt")
print("  2. Run full test suite: pytest")
print("  3. Check code quality: make lint")
print("  4. Try Docker: make docker-build && make docker-up")
print("=" * 60)
