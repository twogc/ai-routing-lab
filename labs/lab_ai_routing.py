"""
Laboratory work: AI-assisted route optimization.

Objective: Learn to use AI agents for automatic optimization of route selection
based on ML predictions of latency and jitter.

Tasks:
1. Create an AI agent for analyzing metrics from Prometheus
2. Integrate the agent with ML prediction models
3. Implement automatic selection of optimal route
4. Evaluate effectiveness of AI-assisted routing
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional

try:
    from cai.sdk.agents import Agent, OpenAIChatCompletionsModel, Runner
    from cai.tools.reconnaissance.generic_linux_command import generic_linux_command

    CAI_AVAILABLE = True
except ImportError:
    CAI_AVAILABLE = False
    logging.warning("CAI framework not available. Install with: pip install cai-framework")

import numpy as np
from dotenv import load_dotenv

from models.prediction.route_prediction_ensemble import RoutePredictionEnsemble

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIRoutingAgent:
    """
    AI agent for route optimization based on ML predictions.

    The agent analyzes network metrics, uses ML models to predict latency
    and jitter, and selects the optimal route.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        prometheus_url: str = "http://localhost:9090",
    ):
        """
        Initialize AI agent for routing.

        Args:
            model_name: LLM model name to use (default from env)
            prometheus_url: Prometheus server URL for metrics collection
        """
        if not CAI_AVAILABLE:
            raise ImportError("CAI framework is required. Install with: pip install cai-framework")

        self.prometheus_url = prometheus_url
        self.model_name = model_name or os.getenv("CAI_MODEL", "openai/gpt-4o")

        # Initialize CAI agent
        self.agent = Agent(
            name="Route Optimizer Agent",
            description="AI agent for network route optimization using ML predictions",
            instructions=self._get_agent_instructions(),
            tools=[generic_linux_command],
            model=OpenAIChatCompletionsModel(
                model=self.model_name,
                openai_client=None,  # Will be initialized automatically
            ),
        )

        # ML model for prediction (will be loaded when needed)
        self.ml_ensemble: Optional[RoutePredictionEnsemble] = None

    def _get_agent_instructions(self) -> str:
        """Get instructions for AI agent."""
        return """
        You are an AI agent specialized in network route optimization.

        Your tasks:
        1. Analyze network metrics from Prometheus
        2. Use ML models to predict latency and jitter for different routes
        3. Select the optimal route based on predictions
        4. Monitor route performance and adapt to changing conditions

        When analyzing routes, consider:
        - Predicted latency (lower is better)
        - Predicted jitter (lower is better)
        - Historical performance data
        - Current network conditions

        Provide clear reasoning for your route selection decisions.
        """

    async def collect_metrics(self) -> Dict[str, float]:
        """
        Collect metrics from Prometheus.

        Returns:
            Dictionary with network metrics (latency, jitter, throughput, etc.)
        """
        logger.info(f"Collecting metrics from {self.prometheus_url}")

        # Use CAI agent to collect metrics
        query = f"""
        Collect network metrics from Prometheus at {self.prometheus_url}.
        Query the following metrics:
        - quic_latency_ms
        - quic_rtt_ms
        - quic_jitter_ms
        - quic_throughput_mbps
        - quic_packet_loss_rate

        Return the current values for these metrics.
        """

        result = await Runner.run(self.agent, query)

        # In real implementation, this would parse agent response
        # and query Prometheus API
        metrics = {
            "latency_ms": 25.5,
            "jitter_ms": 2.3,
            "throughput_mbps": 100.0,
            "packet_loss_rate": 0.01,
        }

        logger.info(f"Collected metrics: {metrics}")
        return metrics

    def predict_route_performance(
        self, route_features: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict route performance using ML model.

        Args:
            route_features: Dictionary with features for each route

        Returns:
            Dictionary with predictions for each route
        """
        if self.ml_ensemble is None:
            logger.warning("ML ensemble not loaded, using default predictions")
            return self._default_predictions(route_features)

        predictions = {}
        for route_id, features in route_features.items():
            features_array = np.array(features).reshape(1, -1)
            prediction = self.ml_ensemble.predict(features_array)

            predictions[route_id] = {
                "predicted_latency_ms": float(prediction.predicted_latency_ms),
                "predicted_jitter_ms": float(prediction.predicted_jitter_ms),
                "confidence_score": float(prediction.confidence_score),
            }

        return predictions

    def _default_predictions(
        self, route_features: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """Generate default predictions (for demonstration)."""
        predictions = {}
        for route_id, features in route_features.items():
            # Simple heuristic based on features
            base_latency = sum(features[:2]) * 10
            base_jitter = features[1] * 2 if len(features) > 1 else 2.0

            predictions[route_id] = {
                "predicted_latency_ms": base_latency,
                "predicted_jitter_ms": base_jitter,
                "confidence_score": 0.85,
            }

        return predictions

    async def select_optimal_route(
        self, route_features: Dict[str, List[float]]
    ) -> tuple[str, Dict[str, Dict[str, float]]]:
        """
        Select optimal route using AI agent.

        Args:
            route_features: Dictionary with features for each route

        Returns:
            Tuple (best_route_id, predictions_dict)
        """
        logger.info(f"Selecting optimal route from {len(route_features)} routes")

        # Get ML predictions
        ml_predictions = self.predict_route_performance(route_features)

        # Use AI agent for final selection
        predictions_summary = "\n".join(
            [
                f"Route {route_id}: latency={pred['predicted_latency_ms']:.2f}ms, "
                f"jitter={pred['predicted_jitter_ms']:.2f}ms, "
                f"confidence={pred['confidence_score']:.2f}"
                for route_id, pred in ml_predictions.items()
            ]
        )

        query = f"""
        Based on the following ML predictions for network routes, select the optimal route:

        {predictions_summary}

        Consider:
        - Lower latency is better
        - Lower jitter is better
        - Higher confidence is better
        - Balance between latency and jitter (70% latency, 30% jitter weight)

        Provide your reasoning and select the best route.
        """

        result = await Runner.run(self.agent, query)

        # Extract selected route from agent response
        # In real implementation, this would parse agent response
        best_route = min(
            ml_predictions.keys(),
            key=lambda r: (
                0.7 * ml_predictions[r]["predicted_latency_ms"]
                + 0.3 * ml_predictions[r]["predicted_jitter_ms"]
            ),
        )

        logger.info(f"Selected optimal route: {best_route}")
        return best_route, ml_predictions

    async def optimize_routing_workflow(self, route_features: Dict[str, List[float]]) -> Dict:
        """
        Complete routing optimization workflow.

        Args:
            route_features: Dictionary with features for each route

        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting routing optimization workflow")

        # Step 1: Collect current metrics
        current_metrics = await self.collect_metrics()

        # Step 2: Select optimal route
        best_route, predictions = await self.select_optimal_route(route_features)

        # Step 3: Generate report
        result = {
            "best_route": best_route,
            "predictions": predictions,
            "current_metrics": current_metrics,
            "optimization_timestamp": asyncio.get_event_loop().time(),
        }

        logger.info(f"Optimization complete. Best route: {best_route}")
        return result


async def main():
    """
    Main function to demonstrate AI routing agent.

    Usage example:
        python -m labs.lab_ai_routing
    """
    if not CAI_AVAILABLE:
        logger.error("CAI framework is not available. Install with: pip install cai-framework")
        return

    # Initialize agent
    agent = AIRoutingAgent()

    # Example route data
    route_features = {
        "route_0": [25.5, 2.3, 0.95, 1.0],
        "route_1": [30.1, 3.1, 0.85, 1.2],
        "route_2": [20.3, 1.8, 0.98, 0.8],
    }

    # Run optimization
    result = await agent.optimize_routing_workflow(route_features)

    # Print results
    print("\n" + "=" * 60)
    print("AI Routing Optimization Results")
    print("=" * 60)
    print(f"Best Route: {result['best_route']}")
    print("\nPredictions:")
    for route_id, pred in result["predictions"].items():
        print(
            f"  {route_id}: latency={pred['predicted_latency_ms']:.2f}ms, "
            f"jitter={pred['predicted_jitter_ms']:.2f}ms"
        )
    print("\nCurrent Metrics:")
    for metric, value in result["current_metrics"].items():
        print(f"  {metric}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
