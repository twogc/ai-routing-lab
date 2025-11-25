"""
Laboratory work: Autonomous Network Orchestration.

Objective: Create a master AI agent (Orchestrator) that coordinates network testing
and VPN management to maintain optimal system performance.

Tasks:
1. Initialize the OrchestratorAgent with BOTH testing and management tools.
2. Execute a closed-loop optimization cycle:
   - Phase 1: Run active tests to assess current performance.
   - Phase 2: Analyze results and decide on mitigation (e.g., block client, change config).
   - Phase 3: Execute mitigation via VPN API.
   - Phase 4: Verify improvement with regression testing.
"""

import asyncio
import logging
import os
from typing import Optional

try:
    from cai.sdk.agents import Agent, OpenAIChatCompletionsModel, Runner

    # Import tools from both domains
    from labs.tools.cai_masque_tools import (
        list_vpn_clients,
        create_vpn_client,
        delete_vpn_client,
        get_vpn_metrics,
    )
    from labs.tools.cai_quic_tools import run_quic_network_test

    CAI_AVAILABLE = True
except ImportError:
    CAI_AVAILABLE = False
    logging.warning("CAI framework not available. Install with: pip install cai-framework")

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Master AI Agent for network orchestration.
    """

    def __init__(self, model_name: Optional[str] = None):
        if not CAI_AVAILABLE:
            raise ImportError("CAI framework is required.")

        self.model_name = model_name or os.getenv("CAI_MODEL", "openai/gpt-4o")

        self.agent = Agent(
            name="Network Orchestrator",
            description="Autonomous orchestrator for VPN optimization and testing",
            instructions=self._get_instructions(),
            tools=[
                # VPN Management
                list_vpn_clients,
                create_vpn_client,
                delete_vpn_client,
                get_vpn_metrics,
                # Active Testing
                run_quic_network_test,
            ],
            model=OpenAIChatCompletionsModel(
                model=self.model_name,
                openai_client=None,
            ),
        )

    def _get_instructions(self) -> str:
        return """
        You are the Chief Network Orchestrator. You have full control over the testing infrastructure and the VPN server.

        Your Goal: Ensure high availability and performance of the VPN service.

        **Workflow:**
        1. **Assess**: Run a quick network test (`run_quic_network_test`) to check current health.
        2. **Monitor**: Check VPN metrics (`get_vpn_metrics`) to see if any specific client is causing load.
        3. **Decide**: 
           - If performance is poor AND a specific client has high errors/load -> Revoke that client.
           - If performance is poor but no specific culprit -> Report "Infrastructure Congestion".
           - If performance is good -> Report "System Healthy".
        4. **Verify**: If you took action, run the test again to confirm improvement.

        **Rules:**
        - Be conservative with `delete_vpn_client`. Only use it if you have evidence (high errors/load).
        - Always explain your plan before executing it.
        """

    async def run_optimization_cycle(self):
        """
        Run a full optimization cycle.
        """
        logger.info("Starting Orchestration Cycle...")

        query = """
        Execute a network optimization cycle:
        1. Run a 'good' profile test for 5s to establish baseline.
        2. Check current VPN metrics.
        3. If you see any client with > 1000 bytes_rx (just as an example threshold for this lab) AND the test showed latency > 100ms, consider them a 'noisy neighbor' and simulate revoking them (or actually revoke if you are sure).
        4. Summarize the state of the network.
        """

        result = await Runner.run(self.agent, query)
        print("\n" + "=" * 60)
        print("Orchestration Report")
        print("=" * 60)
        print(result)
        print("=" * 60)


async def main():
    if not CAI_AVAILABLE:
        logger.error("CAI framework not available.")
        return

    orchestrator = OrchestratorAgent()

    # Run the cycle
    await orchestrator.run_optimization_cycle()


if __name__ == "__main__":
    asyncio.run(main())
