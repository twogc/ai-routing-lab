import json
import logging
import os
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class QuicTestRunner:
    """
    Wrapper for running quic-test binary.
    """

    def __init__(self, binary_path: Optional[str] = None):
        """
        Initialize runner.

        Args:
            binary_path: Path to quic-test binary. If None, tries to find it.
        """
        self.binary_path = binary_path or os.getenv("QUIC_TEST_BINARY", "quic-test")

        # Try to resolve relative path if default is used and we are in the lab repo
        if self.binary_path == "quic-test" and not self._is_executable(self.binary_path):
            # Assumption: quic-test repo is parallel to ai-routing-lab
            potential_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../../../quic-test/bin/quic-test")
            )
            if os.path.exists(potential_path):
                self.binary_path = potential_path

    def _is_executable(self, path: str) -> bool:
        return os.path.isfile(path) and os.access(path, os.X_OK)

    def run_test(
        self,
        mode: str = "test",
        network_profile: str = "excellent",
        duration: str = "10s",
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run quic-test.

        Args:
            mode: 'test', 'client', or 'server'
            network_profile: 'excellent', 'good', 'poor', 'mobile', 'satellite', 'adversarial'
            duration: Duration string (e.g. '10s', '1m')
            extra_args: Additional command line arguments

        Returns:
            Dictionary with execution results (stdout, stderr, exit_code)
        """
        if not self.binary_path:
            return {"error": "quic-test binary not found"}

        cmd = [
            self.binary_path,
            f"--mode={mode}",
            f"--network-profile={network_profile}",
            f"--duration={duration}",
            "--json",  # Force JSON output if supported by binary for easier parsing, or we parse stdout
        ]

        if extra_args:
            cmd.extend(extra_args)

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=float(duration.replace("s", "")) + 5 if "s" in duration else 60,
            )

            output = {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
            }

            # Try to parse last line as JSON if possible, or just return raw
            return output

        except subprocess.TimeoutExpired:
            return {"error": "Test timed out"}
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}
