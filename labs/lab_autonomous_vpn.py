"""
Laboratory work: Autonomous VPN Administrator.

Objective: Create an AI agent that autonomously manages a Masque VPN server,
monitoring metrics and handling client lifecycle to ensure security and performance.

Tasks:
1. Initialize the MasqueAdminAgent with VPN management tools.
2. Monitor VPN metrics for anomalies (e.g., high packet loss, crypto errors).
3. Automatically revoke access for suspicious clients.
4. Report actions and reasoning.
"""

import asyncio
import logging
import os
from typing import Optional

try:
    from cai.sdk.agents import Agent, OpenAIChatCompletionsModel, Runner
    from labs.tools.cai_masque_tools import (
        list_vpn_clients,
        create_vpn_client,
        delete_vpn_client,
        get_vpn_metrics,
    )

    CAI_AVAILABLE = True
except ImportError:
    CAI_AVAILABLE = False
    logging.warning("CAI framework not available. Install with: pip install cai-framework")

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MasqueAdminAgent:
    """
    AI Agent for autonomous VPN administration.
    """

    def __init__(self, model_name: Optional[str] = None):
        if not CAI_AVAILABLE:
            raise ImportError("CAI framework is required.")

        self.model_name = model_name or os.getenv("CAI_MODEL", "openai/gpt-4o")

        self.agent = Agent(
            name="VPN Admin Agent",
            description="Autonomous administrator for Masque VPN",
            instructions=self._get_instructions(),
            tools=[
                list_vpn_clients,
                create_vpn_client,
                delete_vpn_client,
                get_vpn_metrics,
            ],
            model=OpenAIChatCompletionsModel(
                model=self.model_name,
                openai_client=None,
            ),
        )

    def _get_instructions(self) -> str:
        return """
        You are an Autonomous VPN Administrator responsible for the security and performance of a Masque VPN server.

        Your capabilities:
        1. **Monitor System Health**: Check metrics using `get_vpn_metrics`. Look for high packet loss, crypto errors, or unusual traffic patterns.
        2. **Manage Clients**: 
           - List clients with `list_vpn_clients`.
           - Create new clients with `create_vpn_client` if requested.
           - Revoke (delete) clients with `delete_vpn_client` if they pose a security threat or violate policies.

        **Security Policy:**
        - If you detect `quic_crypto_errors` > 10 for a specific client, consider it a potential attack and revoke the client.
        - If `quic_packet_loss_rate` is consistently > 20%, investigate and report.
        - Maintain a list of active clients and ensure no unauthorized access.

        **Protocol:**
        - Always explain your reasoning before taking destructive actions (like deleting a client).
        - Report the status of the system clearly.
        """

    async def run_security_audit(self):
        """
        Run a security audit cycle.
        """
        logger.info("Starting security audit...")

        query = """
        Perform a security audit of the VPN server:
        1. Get the current metrics.
        2. List all active clients.
        3. Analyze if there are any signs of attacks (crypto errors) or performance issues.
        4. If you find a client with > 10 crypto errors, revoke them immediately.
        5. Provide a summary report of the system status.
        """

        result = await Runner.run(self.agent, query)
        print("\n" + "=" * 60)
        print("VPN Security Audit Report")
        print("=" * 60)
        print(result)
        print("=" * 60)


async def main():
    if not CAI_AVAILABLE:
        logger.error("CAI framework not available.")
        return

    admin_agent = MasqueAdminAgent()

    # Run the audit
    await admin_agent.run_security_audit()


if __name__ == "__main__":
    asyncio.run(main())
