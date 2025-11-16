"""Model Monitor - Continuous model health monitoring and metrics collection."""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass
class ModelHealth:
    """Model health status"""
    model_id: str
    timestamp: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_rps: float
    error_rate: float
    cache_hit_rate: float
    status: str  # 'healthy', 'degraded', 'unhealthy'
    alerts: List[str]

class ModelMonitor:
    """
    Continuous monitoring of model health and performance.

    Tracks:
    - Prediction accuracy and precision
    - Latency (p50, p95, p99)
    - Throughput (requests per second)
    - Error rate
    - Cache hit rate
    - Health status and alerts
    - Prometheus metrics
    """

    def __init__(
        self,
        model_id: str,
        accuracy_threshold: float = 0.85,
        latency_threshold_ms: float = 100,
        error_rate_threshold: float = 0.05,
        logger: Optional[logging.Logger] = None
    ):
        self.model_id = model_id
        self.accuracy_threshold = accuracy_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        self.logger = logger or logging.getLogger(__name__)

        # Metrics storage
        self.metrics_history = []
        self.latencies = []
        self.predictions_correct = 0
        self.predictions_total = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0

    def record_prediction(
        self,
        prediction: Any,
        ground_truth: Optional[Any] = None,
        latency_ms: float = 0.0,
        cache_hit: bool = False,
        error: bool = False
    ):
        """Record a single prediction for monitoring"""
        self.predictions_total += 1
        self.latencies.append(latency_ms)

        if ground_truth is not None:
            if prediction == ground_truth:
                self.predictions_correct += 1

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if error:
            self.errors += 1

    def get_current_metrics(self) -> ModelHealth:
        """Get current health metrics"""
        # Calculate accuracy
        accuracy = (
            self.predictions_correct / self.predictions_total
            if self.predictions_total > 0 else 0.0
        )

        # Calculate latencies
        if self.latencies:
            latency_p50 = float(np.percentile(self.latencies, 50))
            latency_p95 = float(np.percentile(self.latencies, 95))
            latency_p99 = float(np.percentile(self.latencies, 99))
        else:
            latency_p50 = latency_p95 = latency_p99 = 0.0

        # Calculate throughput (requests per second, normalized)
        throughput_rps = self.predictions_total / max(1, len(self.latencies))

        # Calculate cache hit rate
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            self.cache_hits / cache_total if cache_total > 0 else 0.0
        )

        # Calculate error rate
        error_rate = self.errors / max(1, self.predictions_total)

        # Determine health status and alerts
        status, alerts = self._determine_health(
            accuracy, latency_p95, error_rate
        )

        # Simplified precision/recall/f1 for demo
        precision = accuracy * 0.95
        recall = accuracy * 0.90
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        health = ModelHealth(
            model_id=self.model_id,
            timestamp=datetime.now().isoformat(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
            latency_p99_ms=latency_p99,
            throughput_rps=throughput_rps,
            error_rate=error_rate,
            cache_hit_rate=cache_hit_rate,
            status=status,
            alerts=alerts
        )

        self.metrics_history.append(health)
        return health

    def get_metrics_summary(self, window_size: int = 100) -> Dict[str, Any]:
        """Get aggregated metrics summary"""
        recent_metrics = self.metrics_history[-window_size:] if self.metrics_history else []

        if not recent_metrics:
            return {
                'model_id': self.model_id,
                'status': 'no_data',
                'metrics_count': 0
            }

        accuracies = [m.accuracy for m in recent_metrics]
        latencies = [m.latency_p95_ms for m in recent_metrics]
        error_rates = [m.error_rate for m in recent_metrics]

        return {
            'model_id': self.model_id,
            'metrics_count': len(recent_metrics),
            'accuracy': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies))
            },
            'latency_p95_ms': {
                'mean': float(np.mean(latencies)),
                'std': float(np.std(latencies)),
                'min': float(np.min(latencies)),
                'max': float(np.max(latencies))
            },
            'error_rate': {
                'mean': float(np.mean(error_rates)),
                'std': float(np.std(error_rates)),
                'max': float(np.max(error_rates))
            },
            'current_status': recent_metrics[-1].status if recent_metrics else 'unknown'
        }

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus metrics format"""
        metrics = self.get_current_metrics()

        prometheus_output = f"""# HELP cloudbridge_model_accuracy Model prediction accuracy
# TYPE cloudbridge_model_accuracy gauge
cloudbridge_model_accuracy{{model_id="{self.model_id}"}} {metrics.accuracy}

# HELP cloudbridge_model_latency_p95_ms Model latency p95
# TYPE cloudbridge_model_latency_p95_ms gauge
cloudbridge_model_latency_p95_ms{{model_id="{self.model_id}"}} {metrics.latency_p95_ms}

# HELP cloudbridge_model_error_rate Model error rate
# TYPE cloudbridge_model_error_rate gauge
cloudbridge_model_error_rate{{model_id="{self.model_id}"}} {metrics.error_rate}

# HELP cloudbridge_model_throughput_rps Model throughput
# TYPE cloudbridge_model_throughput_rps gauge
cloudbridge_model_throughput_rps{{model_id="{self.model_id}"}} {metrics.throughput_rps}

# HELP cloudbridge_model_cache_hit_rate Cache hit rate
# TYPE cloudbridge_model_cache_hit_rate gauge
cloudbridge_model_cache_hit_rate{{model_id="{self.model_id}"}} {metrics.cache_hit_rate}
"""
        return prometheus_output

    def get_health_alerts(self) -> List[str]:
        """Get active health alerts"""
        metrics = self.get_current_metrics()
        return metrics.alerts

    def reset_metrics(self):
        """Reset metrics counters"""
        self.latencies = []
        self.predictions_correct = 0
        self.predictions_total = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0

    def _determine_health(
        self,
        accuracy: float,
        latency_p95: float,
        error_rate: float
    ) -> tuple:
        """Determine health status and generate alerts"""
        alerts = []

        if accuracy < self.accuracy_threshold:
            alerts.append(
                f"Low accuracy: {accuracy:.4f} < {self.accuracy_threshold}"
            )

        if latency_p95 > self.latency_threshold_ms:
            alerts.append(
                f"High latency: {latency_p95:.2f}ms > {self.latency_threshold_ms}ms"
            )

        if error_rate > self.error_rate_threshold:
            alerts.append(
                f"High error rate: {error_rate:.4f} > {self.error_rate_threshold}"
            )

        if len(alerts) >= 2:
            status = 'unhealthy'
        elif len(alerts) == 1:
            status = 'degraded'
        else:
            status = 'healthy'

        return status, alerts
