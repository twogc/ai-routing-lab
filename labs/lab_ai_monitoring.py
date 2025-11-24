"""
Laboratory work: AI monitoring of network infrastructure.

Objective: Learn to use AI agents for automated monitoring and analysis
of network infrastructure metrics, anomaly detection, and alerting.

Tasks:
1. Create AI agent for analyzing Prometheus metrics
2. Implement automated anomaly detection
3. Set up intelligent alerting based on AI analysis
4. Integrate with existing monitoring infrastructure
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional

try:
    from cai.sdk.agents import Agent, Runner, OpenAIChatCompletionsModel
    from cai.tools.reconnaissance.generic_linux_command import generic_linux_command
    CAI_AVAILABLE = True
except ImportError:
    CAI_AVAILABLE = False
    logging.warning(
        "CAI framework not available. Install with: pip install cai-framework"
    )

from dotenv import load_dotenv

from models.anomaly.isolation_forest import IsolationForestModel
import numpy as np

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIMonitoringAgent:
    """
    AI agent for monitoring network infrastructure.

    Analyzes Prometheus metrics, detects anomalies, and provides
    intelligent insights about network performance.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        prometheus_url: str = "http://localhost:9090",
    ):
        """
        Initialize AI monitoring agent.

        Args:
            model_name: LLM model name to use (default from env)
            prometheus_url: Prometheus server URL
        """
        if not CAI_AVAILABLE:
            raise ImportError(
                "CAI framework is required. Install with: pip install cai-framework"
            )

        self.prometheus_url = prometheus_url
        self.model_name = model_name or os.getenv("CAI_MODEL", "openai/gpt-4o")

        # Initialize CAI agent
        self.agent = Agent(
            name="Network Monitoring Agent",
            description="AI agent for monitoring and analyzing network infrastructure",
            instructions=self._get_agent_instructions(),
            tools=[generic_linux_command],
            model=OpenAIChatCompletionsModel(
                model=self.model_name,
                openai_client=None,
            ),
        )

        # Anomaly detection model
        self.anomaly_detector: Optional[IsolationForestModel] = None

    def _get_agent_instructions(self) -> str:
        """Get instructions for monitoring agent."""
        return """
        You are an AI agent specialized in network infrastructure monitoring.

        Your tasks:
        1. Analyze metrics from Prometheus
        2. Identify patterns and trends in network performance
        3. Detect anomalies and unusual behavior
        4. Provide insights and recommendations
        5. Generate alerts for critical issues

        When analyzing metrics, consider:
        - Latency trends and spikes
        - Jitter variations
        - Throughput patterns
        - Packet loss rates
        - Resource utilization

        Provide clear, actionable insights.
        """

    async def collect_and_analyze_metrics(self) -> Dict[str, any]:
        """
        Collect metrics from Prometheus and analyze them.

        Returns:
            Dictionary with metrics and analysis
        """
        logger.info(f"Collecting and analyzing metrics from {self.prometheus_url}")

        query = f"""
        Collect and analyze network metrics from Prometheus at {self.prometheus_url}.

        Query the following metrics:
        - quic_latency_ms
        - quic_rtt_ms
        - quic_jitter_ms
        - quic_throughput_mbps
        - quic_packet_loss_rate

        Analyze:
        1. Current values vs historical averages
        2. Trends over time
        3. Potential issues or anomalies
        4. Performance recommendations

        Provide a comprehensive analysis.
        """

        result = await Runner.run(self.agent, query)

        # In real implementation, this would parse agent response
        # and query Prometheus API
        metrics = {
            "latency_ms": 25.5,
            "jitter_ms": 2.3,
            "throughput_mbps": 100.0,
            "packet_loss_rate": 0.01,
            "analysis": str(result),
        }

        logger.info("Metrics collected and analyzed")
        return metrics

    def detect_anomalies(self, metrics_data: List[Dict[str, float]]) -> Dict[str, any]:
        """
        Detect anomalies in metrics data using ML model.

        Args:
            metrics_data: List of metric dictionaries

        Returns:
            Dictionary with anomaly detection results
        """
        if not metrics_data:
            return {"anomalies": [], "status": "no_data"}

        # Prepare data for anomaly detection
        features = []
        for metric in metrics_data:
            feature_vector = [
                metric.get("latency_ms", 0),
                metric.get("jitter_ms", 0),
                metric.get("throughput_mbps", 0),
                metric.get("packet_loss_rate", 0),
            ]
            features.append(feature_vector)

        X = np.array(features)

        # Initialize and train anomaly detector if needed
        if self.anomaly_detector is None:
            logger.info("Training anomaly detection model")
            self.anomaly_detector = IsolationForestModel(
                n_trees=100,
                contamination=0.1,
            )
            self.anomaly_detector.fit(X)

        # Detect anomalies
        predictions, scores = self.anomaly_detector.predict(X)

        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == 1:  # Anomaly detected
                anomalies.append({
                    "index": i,
                    "metrics": metrics_data[i],
                    "anomaly_score": float(score),
                })

        return {
            "anomalies": anomalies,
            "total_samples": len(metrics_data),
            "anomaly_count": len(anomalies),
            "status": "completed",
        }

    async def generate_insights(self, metrics: Dict[str, any]) -> Dict[str, any]:
        """
        Generate AI-powered insights from metrics.

        Args:
            metrics: Dictionary with collected metrics

        Returns:
            Dictionary with insights and recommendations
        """
        logger.info("Generating AI insights")

        metrics_summary = f"""
        Current Network Metrics:
        - Latency: {metrics.get('latency_ms', 'N/A')} ms
        - Jitter: {metrics.get('jitter_ms', 'N/A')} ms
        - Throughput: {metrics.get('throughput_mbps', 'N/A')} Mbps
        - Packet Loss: {metrics.get('packet_loss_rate', 'N/A')} %
        """

        query = f"""
        Based on the following network metrics, provide insights and recommendations:

        {metrics_summary}

        Analyze:
        1. Overall network health
        2. Performance trends
        3. Potential bottlenecks
        4. Optimization opportunities
        5. Actionable recommendations

        Provide detailed insights.
        """

        result = await Runner.run(self.agent, query)

        return {
            "insights": str(result),
            "metrics": metrics,
            "timestamp": asyncio.get_event_loop().time(),
        }

    async def monitor_workflow(self) -> Dict[str, any]:
        """
        Complete monitoring workflow.

        Returns:
            Dictionary with monitoring results
        """
        logger.info("Starting monitoring workflow")

        # Step 1: Collect and analyze metrics
        metrics = await self.collect_and_analyze_metrics()

        # Step 2: Detect anomalies (using sample data)
        sample_metrics = [metrics] * 10  # Simulate multiple samples
        anomaly_results = self.detect_anomalies(sample_metrics)

        # Step 3: Generate insights
        insights = await self.generate_insights(metrics)

        # Step 4: Compile results
        result = {
            "metrics": metrics,
            "anomaly_detection": anomaly_results,
            "insights": insights,
            "status": "completed",
        }

        logger.info("Monitoring workflow complete")
        return result


async def main():
    """
    Main function to demonstrate AI monitoring.

    Usage example:
        python -m labs.lab_ai_monitoring
    """
    if not CAI_AVAILABLE:
        logger.error(
            "CAI framework is not available. Install with: pip install cai-framework"
        )
        return

    # Initialize monitoring agent
    agent = AIMonitoringAgent()

    # Run monitoring workflow
    results = await agent.monitor_workflow()

    # Print results
    print("\n" + "=" * 60)
    print("AI Monitoring Results")
    print("=" * 60)

    print("\nMetrics:")
    for key, value in results["metrics"].items():
        if key != "analysis":
            print(f"  {key}: {value}")

    print("\nAnomaly Detection:")
    print(f"  Total samples: {results['anomaly_detection']['total_samples']}")
    print(f"  Anomalies detected: {results['anomaly_detection']['anomaly_count']}")

    print("\nInsights:")
    print(f"  {results['insights']['insights'][:200]}...")


if __name__ == "__main__":
    asyncio.run(main())

