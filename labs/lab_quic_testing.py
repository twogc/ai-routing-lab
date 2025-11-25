"""
Laboratory work: Autonomous QUIC Protocol Testing.

Objective: Create an AI agent that actively tests the QUIC protocol under various
network conditions to identify performance bottlenecks and stability issues.

Tasks:
1. Initialize the QuicTestAgent with network testing tools.
2. Run tests with different network profiles (mobile, satellite).
3. Analyze the results to find which profile causes the most packet loss or latency.
4. Generate a report on protocol stability.
"""

import asyncio
import logging
import os
from typing import Optional

try:
    from cai.sdk.agents import Agent, OpenAIChatCompletionsModel, Runner
    from labs.tools.cai_quic_tools import run_quic_network_test

    CAI_AVAILABLE = True
except ImportError:
    CAI_AVAILABLE = False
    logging.warning("CAI framework not available. Install with: pip install cai-framework")

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuicTestAgent:
    """
    AI Agent for active QUIC protocol testing.
    """

    def __init__(self, model_name: Optional[str] = None):
        if not CAI_AVAILABLE:
            raise ImportError("CAI framework is required.")

        self.model_name = model_name or os.getenv("CAI_MODEL", "openai/gpt-4o")

        self.agent = Agent(
            name="QUIC Test Agent",
            description="Autonomous QA agent for network protocol testing",
            instructions=self._get_instructions(),
            tools=[run_quic_network_test],
            model=OpenAIChatCompletionsModel(
                model=self.model_name,
                openai_client=None,
            ),
        )

    def _get_instructions(self) -> str:
        return """
        You are a Network QA Engineer specialized in the QUIC protocol.

        Your goal is to stress-test the QUIC implementation and find its breaking points.

        Your capabilities:
        1. **Run Tests**: Use `run_quic_network_test` to execute the `quic-test` binary.
           - You can specify profiles: 'excellent', 'good', 'poor', 'mobile', 'satellite', 'adversarial'.
           - You can specify duration.

        **Workflow:**
        1. Start with a baseline test ('excellent' profile).
        2. Progressively test harder profiles ('mobile', 'satellite').
        3. Analyze the output of each test. Look for:
           - High latency (> 500ms)
           - Packet loss (> 5%)
           - Connection failures
        4. If a test shows poor performance, explain WHY based on the metrics (e.g., "Satellite profile caused high latency due to physical distance simulation").
        5. Summarize which profile is the most challenging for the current implementation.
        """

    async def run_stress_test(self):
        """
        Run a full stress test suite.
        """
        logger.info("Starting QUIC stress test...")
        
        query = """
        Conduct a stress test of the QUIC protocol.
        1. Run a short 5s test on 'excellent' profile to establish baseline.
        2. Run a 5s test on 'mobile' profile.
        3. Run a 5s test on 'satellite' profile.
        4. Compare the results. Which profile had the worst performance?
        5. Provide a final stability report.
        """
        
        result = await Runner.run(self.agent, query)
        print("\n" + "="*60)
        print("QUIC Stress Test Report")
        print("="*60)
        print(result)
        print("="*60)

async def main():
    if not CAI_AVAILABLE:
        logger.error("CAI framework not available.")
        return

    tester = QuicTestAgent()
    
    # Run the stress test
    await tester.run_stress_test()

if __name__ == "__main__":
    asyncio.run(main())
