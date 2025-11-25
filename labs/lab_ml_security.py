"""
Laboratory work: ML system security testing.

Objective: Learn to use CAI framework for testing security of ML systems,
including API security, model security, and infrastructure security.

Tasks:
1. Test FastAPI service for security vulnerabilities
2. Analyze ML model security (adversarial attacks, model poisoning)
3. Test Model Registry security
4. Evaluate infrastructure security (Docker, Prometheus, etc.)
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional

try:
    from cai.sdk.agents import Agent, OpenAIChatCompletionsModel, Runner
    from cai.tools.reconnaissance.curl import curl_request
    from cai.tools.reconnaissance.generic_linux_command import generic_linux_command

    CAI_AVAILABLE = True
except ImportError:
    CAI_AVAILABLE = False
    logging.warning("CAI framework not available. Install with: pip install cai-framework")

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLSecurityTester:
    """
    Security testing agent for ML systems using CAI framework.

    Tests FastAPI endpoints, ML models, and infrastructure components
    for security vulnerabilities.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_url: str = "http://localhost:5000",
    ):
        """
        Initialize ML security tester.

        Args:
            model_name: LLM model name to use (default from env)
            api_url: FastAPI service URL to test
        """
        if not CAI_AVAILABLE:
            raise ImportError("CAI framework is required. Install with: pip install cai-framework")

        self.api_url = api_url
        self.model_name = model_name or os.getenv("CAI_MODEL", "openai/gpt-4o")

        # Initialize CAI agent for security testing
        self.agent = Agent(
            name="ML Security Tester",
            description="AI agent for testing security of ML systems and APIs",
            instructions=self._get_agent_instructions(),
            tools=[curl_request, generic_linux_command],
            model=OpenAIChatCompletionsModel(
                model=self.model_name,
                openai_client=None,
            ),
        )

    def _get_agent_instructions(self) -> str:
        """Get instructions for security testing agent."""
        return """
        You are a security expert testing ML systems and APIs.

        Your tasks:
        1. Test FastAPI endpoints for common vulnerabilities:
           - SQL injection
           - XSS (Cross-Site Scripting)
           - Input validation issues
           - Rate limiting bypass
           - Authentication/authorization flaws
           - Path traversal
           - Command injection

        2. Analyze ML model security:
           - Adversarial attacks
           - Model poisoning
           - Data leakage
           - Model extraction

        3. Test infrastructure security:
           - Docker container security
           - Prometheus/Grafana security
           - Network security

        Always provide detailed findings and recommendations.
        """

    async def test_api_endpoints(self) -> Dict[str, List[Dict]]:
        """
        Test FastAPI endpoints for security vulnerabilities.

        Returns:
            Dictionary with test results for each endpoint
        """
        logger.info(f"Testing API endpoints at {self.api_url}")

        endpoints = [
            "/health",
            "/predict",
            "/predict/routes",
            "/models",
            "/metrics",
        ]

        results = {}

        for endpoint in endpoints:
            logger.info(f"Testing endpoint: {endpoint}")

            query = f"""
            Test the FastAPI endpoint {self.api_url}{endpoint} for security vulnerabilities.

            Perform the following tests:
            1. Test for SQL injection (if applicable)
            2. Test for XSS vulnerabilities
            3. Test input validation (malformed JSON, oversized payloads)
            4. Test for rate limiting
            5. Test for authentication/authorization
            6. Test for path traversal
            7. Test for command injection

            Provide detailed findings for each test.
            """

            try:
                result = await Runner.run(self.agent, query)
                results[endpoint] = [
                    {
                        "status": "tested",
                        "endpoint": endpoint,
                        "result": str(result),
                    }
                ]
            except Exception as e:
                logger.error(f"Error testing {endpoint}: {e}")
                results[endpoint] = [
                    {
                        "status": "error",
                        "endpoint": endpoint,
                        "error": str(e),
                    }
                ]

        return results

    async def test_model_security(self) -> Dict[str, any]:
        """
        Test ML model security for adversarial attacks and model poisoning.

        Returns:
            Dictionary with model security test results
        """
        logger.info("Testing ML model security")

        query = """
        Analyze the ML models in AI Routing Lab for security vulnerabilities:

        1. Adversarial Attacks:
           - Test if models are vulnerable to adversarial examples
           - Check robustness to input perturbations
           - Evaluate model confidence on edge cases

        2. Model Poisoning:
           - Check if training data validation is in place
           - Evaluate model versioning and integrity checks
           - Test Model Registry security

        3. Model Extraction:
           - Check if model weights are protected
           - Evaluate API rate limiting for model queries
           - Test for model information leakage

        Provide recommendations for improving model security.
        """

        try:
            result = await Runner.run(self.agent, query)
            return {
                "status": "tested",
                "result": str(result),
            }
        except Exception as e:
            logger.error(f"Error testing model security: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def test_infrastructure_security(self) -> Dict[str, any]:
        """
        Test infrastructure security (Docker, Prometheus, etc.).

        Returns:
            Dictionary with infrastructure security test results
        """
        logger.info("Testing infrastructure security")

        query = """
        Test the infrastructure components of AI Routing Lab for security:

        1. Docker Security:
           - Check container configurations
           - Test for privilege escalation
           - Evaluate network isolation
           - Check for exposed ports

        2. Prometheus Security:
           - Test authentication/authorization
           - Check for exposed metrics
           - Evaluate access controls

        3. Network Security:
           - Test inter-container communication
           - Check for exposed services
           - Evaluate firewall rules

        Provide security recommendations.
        """

        try:
            result = await Runner.run(self.agent, query)
            return {
                "status": "tested",
                "result": str(result),
            }
        except Exception as e:
            logger.error(f"Error testing infrastructure security: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def run_full_security_audit(self) -> Dict:
        """
        Run complete security audit of ML system.

        Returns:
            Dictionary with complete security audit results
        """
        logger.info("Starting full security audit")

        results = {
            "api_security": await self.test_api_endpoints(),
            "model_security": await self.test_model_security(),
            "infrastructure_security": await self.test_infrastructure_security(),
        }

        logger.info("Security audit complete")
        return results


async def main():
    """
    Main function to demonstrate ML security testing.

    Usage example:
        python -m labs.lab_ml_security
    """
    if not CAI_AVAILABLE:
        logger.error("CAI framework is not available. Install with: pip install cai-framework")
        return

    # Initialize security tester
    tester = MLSecurityTester()

    # Run full security audit
    results = await tester.run_full_security_audit()

    # Print results
    print("\n" + "=" * 60)
    print("ML System Security Audit Results")
    print("=" * 60)

    print("\nAPI Security Tests:")
    for endpoint, tests in results["api_security"].items():
        print(f"  {endpoint}: {tests[0]['status']}")

    print("\nModel Security:")
    print(f"  Status: {results['model_security']['status']}")

    print("\nInfrastructure Security:")
    print(f"  Status: {results['infrastructure_security']['status']}")


if __name__ == "__main__":
    asyncio.run(main())
