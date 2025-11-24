"""
Examples of CAI framework integration with AI Routing Lab.

This module demonstrates various ways to integrate CAI (Cybersecurity AI)
framework with AI Routing Lab for educational and research purposes.

Examples:
1. Basic AI agent for route optimization
2. Security testing agent for ML systems
3. Monitoring agent for network infrastructure
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

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_agent():
    """
    Example 1: Basic AI agent setup.

    Demonstrates how to create a simple AI agent using CAI framework.
    """
    if not CAI_AVAILABLE:
        logger.error("CAI framework is not available")
        return

    print("\n" + "=" * 60)
    print("Example 1: Basic AI Agent")
    print("=" * 60)

    # Create a basic agent
    agent = Agent(
        name="Basic Agent",
        description="A basic AI agent for demonstration",
        instructions="You are a helpful AI assistant.",
        model=OpenAIChatCompletionsModel(
            model=os.getenv("CAI_MODEL", "openai/gpt-4o"),
            openai_client=None,
        ),
    )

    print("Agent created successfully")
    print(f"Agent name: {agent.name}")
    print(f"Agent description: {agent.description}")


async def example_route_optimization_agent():
    """
    Example 2: Route optimization agent.

    Demonstrates how to create an AI agent for route optimization.
    """
    if not CAI_AVAILABLE:
        logger.error("CAI framework is not available")
        return

    print("\n" + "=" * 60)
    print("Example 2: Route Optimization Agent")
    print("=" * 60)

    # Create route optimization agent
    agent = Agent(
        name="Route Optimizer",
        description="AI agent for network route optimization",
        instructions="""
        You are an AI agent specialized in network route optimization.
        Analyze route metrics and select the optimal route based on
        latency and jitter predictions.
        """,
        tools=[generic_linux_command],
        model=OpenAIChatCompletionsModel(
            model=os.getenv("CAI_MODEL", "openai/gpt-4o"),
            openai_client=None,
        ),
    )

    # Example query
    query = """
    Analyze the following route metrics and select the best route:
    - Route A: latency=25ms, jitter=2ms
    - Route B: latency=30ms, jitter=3ms
    - Route C: latency=20ms, jitter=1.8ms
    """

    print("Running route optimization query...")
    result = await Runner.run(agent, query)
    print(f"Agent response: {result}")


async def example_security_testing_agent():
    """
    Example 3: Security testing agent.

    Demonstrates how to create an AI agent for security testing.
    """
    if not CAI_AVAILABLE:
        logger.error("CAI framework is not available")
        return

    print("\n" + "=" * 60)
    print("Example 3: Security Testing Agent")
    print("=" * 60)

    # Create security testing agent
    agent = Agent(
        name="Security Tester",
        description="AI agent for security testing of ML systems",
        instructions="""
        You are a security expert testing ML systems.
        Test APIs and models for common vulnerabilities:
        - SQL injection
        - XSS vulnerabilities
        - Input validation issues
        - Authentication flaws
        """,
        tools=[generic_linux_command],
        model=OpenAIChatCompletionsModel(
            model=os.getenv("CAI_MODEL", "openai/gpt-4o"),
            openai_client=None,
        ),
    )

    # Example query
    query = """
    Test the FastAPI endpoint http://localhost:5000/predict
    for security vulnerabilities. Focus on input validation.
    """

    print("Running security test query...")
    result = await Runner.run(agent, query)
    print(f"Security test results: {result}")


async def example_monitoring_agent():
    """
    Example 4: Monitoring agent.

    Demonstrates how to create an AI agent for infrastructure monitoring.
    """
    if not CAI_AVAILABLE:
        logger.error("CAI framework is not available")
        return

    print("\n" + "=" * 60)
    print("Example 4: Monitoring Agent")
    print("=" * 60)

    # Create monitoring agent
    agent = Agent(
        name="Network Monitor",
        description="AI agent for monitoring network infrastructure",
        instructions="""
        You are an AI agent specialized in network monitoring.
        Analyze Prometheus metrics, detect anomalies, and provide
        insights about network performance.
        """,
        tools=[generic_linux_command],
        model=OpenAIChatCompletionsModel(
            model=os.getenv("CAI_MODEL", "openai/gpt-4o"),
            openai_client=None,
        ),
    )

    # Example query
    query = """
    Analyze the following network metrics:
    - Latency: 25.5ms
    - Jitter: 2.3ms
    - Throughput: 100Mbps
    - Packet loss: 0.01%

    Provide insights and recommendations.
    """

    print("Running monitoring query...")
    result = await Runner.run(agent, query)
    print(f"Monitoring insights: {result}")


async def main():
    """
    Run all CAI integration examples.

    Usage:
        python examples/cai_integration.py
    """
    print("CAI Framework Integration Examples")
    print("=" * 60)

    # Example 1: Basic agent
    example_basic_agent()

    # Example 2: Route optimization
    if CAI_AVAILABLE:
        await example_route_optimization_agent()

    # Example 3: Security testing
    if CAI_AVAILABLE:
        await example_security_testing_agent()

    # Example 4: Monitoring
    if CAI_AVAILABLE:
        await example_monitoring_agent()

    print("\n" + "=" * 60)
    print("All examples completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

