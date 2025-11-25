import os
from typing import List, Optional

from cai.sdk.agents import function_tool

from labs.tools.quic_runner import QuicTestRunner

runner = QuicTestRunner()


@function_tool
def run_quic_network_test(profile: str, duration: str = "10s") -> str:
    """
    Run a QUIC protocol test under specific network conditions.

    Args:
        profile: Network profile to simulate. Options: 'excellent', 'good', 'poor', 'mobile', 'satellite', 'adversarial'.
        duration: Duration of the test (e.g., '10s', '1m').

    Returns:
        String containing the test results (metrics summary).
    """
    result = runner.run_test(mode="test", network_profile=profile, duration=duration)

    if "error" in result:
        return f"Test failed: {result['error']}"

    # In a real scenario, we would parse the JSON output from stdout
    # For now, we return the raw stdout/stderr
    output = f"Test completed (Exit Code: {result['exit_code']})\n"
    output += f"Command: {result['command']}\n"
    output += f"Output:\n{result['stdout']}\n"
    if result["stderr"]:
        output += f"Errors/Logs:\n{result['stderr']}"

    return output


@function_tool
def analyze_quic_performance(metrics_json: str) -> str:
    """
    Analyze raw QUIC metrics JSON and provide a summary.

    Args:
        metrics_json: Raw JSON string from a test run.

    Returns:
        Analysis summary.
    """
    # This is a helper tool for the agent to "think" about the data
    # In reality, the agent reads the text, but this could be a heuristic function
    return "Analysis not implemented in tool, please analyze the raw text output directly."
