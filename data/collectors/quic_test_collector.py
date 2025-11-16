"""
QUIC Test Metrics Collector

Collects metrics from quic-test tool via Prometheus or JSON export.
Integrates with CloudBridge quic-test for real QUIC traffic data.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import requests
from prometheus_client.parser import text_string_to_metric_families
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class PrometheusCollector:
    """Collect metrics from Prometheus endpoint (exported by quic-test)."""

    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        """
        Initialize Prometheus collector.

        Args:
            prometheus_url: URL of Prometheus instance
        """
        self.prometheus_url = prometheus_url
        self.session = requests.Session()

    def collect_latency_metrics(self, route_id: Optional[str] = None) -> Dict:
        """
        Collect latency metrics from Prometheus.

        Args:
            route_id: Optional route identifier for filtering

        Returns:
            Dictionary with latency metrics (p50, p95, p99, jitter)
        """
        query = "quic_latency_p95_ms"
        if route_id:
            query += f'{{route_id="{route_id}"}}'

        try:
            response = self.session.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()

            metrics = {}
            if data.get("status") == "success" and data.get("data", {}).get("result"):
                for result in data["data"]["result"]:
                    metric_name = result["metric"].get("__name__", "")
                    value = float(result["value"][1])
                    metrics[metric_name] = value

            return metrics
        except Exception as e:
            logger.error(f"Error collecting Prometheus metrics: {e}")
            return {}

    def collect_jitter_metrics(self, route_id: Optional[str] = None) -> Dict:
        """
        Collect jitter metrics from Prometheus.

        Args:
            route_id: Optional route identifier for filtering

        Returns:
            Dictionary with jitter metrics
        """
        query = "quic_jitter_p95_ms"
        if route_id:
            query += f'{{route_id="{route_id}"}}'

        try:
            response = self.session.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()

            metrics = {}
            if data.get("status") == "success" and data.get("data", {}).get("result"):
                for result in data["data"]["result"]:
                    metric_name = result["metric"].get("__name__", "")
                    value = float(result["value"][1])
                    metrics[metric_name] = value

            return metrics
        except Exception as e:
            logger.error(f"Error collecting jitter metrics: {e}")
            return {}

    def collect_all_metrics(self) -> Dict:
        """
        Collect all available metrics from Prometheus.

        Returns:
            Dictionary with all collected metrics
        """
        metrics = {
            "latency": self.collect_latency_metrics(),
            "jitter": self.collect_jitter_metrics(),
            "timestamp": datetime.now().isoformat(),
        }
        return metrics


class JSONFileCollector(FileSystemEventHandler):
    """Collect metrics from JSON files exported by quic-test."""

    def __init__(self, watch_directory: str, output_callback=None):
        """
        Initialize JSON file collector.

        Args:
            watch_directory: Directory to watch for JSON files
            output_callback: Callback function to process collected metrics
        """
        self.watch_directory = Path(watch_directory)
        self.output_callback = output_callback
        self.observer = Observer()

    def on_created(self, event):
        """Handle new file creation."""
        if event.src_path.endswith('.json'):
            self.process_file(event.src_path)

    def process_file(self, file_path: str):
        """
        Process JSON file from quic-test.

        Args:
            file_path: Path to JSON file
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract relevant metrics
            metrics = self.extract_metrics(data)

            if self.output_callback:
                self.output_callback(metrics)
            else:
                logger.info(f"Collected metrics from {file_path}: {metrics}")

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

    def extract_metrics(self, data: Dict) -> Dict:
        """
        Extract latency/jitter metrics from quic-test JSON export.

        Args:
            data: JSON data from quic-test

        Returns:
            Dictionary with extracted metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
        }

        # Extract latency metrics
        if "Latency" in data:
            latency = data["Latency"]
            metrics["latency"] = {
                "p50": latency.get("P50", 0),
                "p95": latency.get("P95", 0),
                "p99": latency.get("P99", 0),
                "average": latency.get("Average", 0),
                "jitter": latency.get("Jitter", 0),
            }

        # Extract throughput metrics
        if "Throughput" in data:
            throughput = data["Throughput"]
            metrics["throughput"] = {
                "average": throughput.get("Average", 0),
                "peak": throughput.get("Peak", 0),
            }

        # Extract route information if available
        if "RouteID" in data:
            metrics["route_id"] = data["RouteID"]
        if "SourcePoP" in data:
            metrics["source_pop"] = data["SourcePoP"]
        if "TargetPoP" in data:
            metrics["target_pop"] = data["TargetPoP"]

        return metrics

    def start_watching(self):
        """Start watching directory for new files."""
        self.observer.schedule(self, str(self.watch_directory), recursive=False)
        self.observer.start()
        logger.info(f"Started watching directory: {self.watch_directory}")

    def stop_watching(self):
        """Stop watching directory."""
        self.observer.stop()
        self.observer.join()


def main():
    """Main entry point for collector."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect metrics from quic-test")
    parser.add_argument(
        "--prometheus-url",
        default="http://localhost:9090",
        help="Prometheus URL"
    )
    parser.add_argument(
        "--watch-dir",
        help="Directory to watch for JSON files"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Output directory for collected metrics"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Collect from Prometheus
    if args.prometheus_url:
        collector = PrometheusCollector(args.prometheus_url)
        metrics = collector.collect_all_metrics()
        logger.info(f"Collected metrics: {metrics}")

    # Watch for JSON files
    if args.watch_dir:
        json_collector = JSONFileCollector(args.watch_dir)
        json_collector.start_watching()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            json_collector.stop_watching()


if __name__ == "__main__":
    main()

