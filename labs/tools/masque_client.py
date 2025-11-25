import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class MasqueVPNClient:
    """
    Client for interacting with Masque VPN HTTP API.
    """

    def __init__(self, base_url: str = "http://localhost:8080", auth: Optional[tuple] = None):
        """
        Initialize Masque VPN client.

        Args:
            base_url: Base URL of the VPN server API (e.g., http://localhost:8080)
            auth: Tuple of (username, password) for Basic Auth, or None
        """
        self.base_url = base_url.rstrip("/")
        self.auth = auth

    def _get_url(self, endpoint: str) -> str:
        return f"{self.base_url}/api{endpoint}"

    def list_clients(self) -> List[Dict[str, Any]]:
        """
        Get list of all registered clients.

        Returns:
            List of client dictionaries.
        """
        try:
            url = self._get_url("/clients")
            response = requests.get(url, auth=self.auth, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to list clients: {e}")
            return []

    def create_client(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Create a new client.

        Args:
            name: Name of the client

        Returns:
            Created client dictionary or None if failed.
        """
        try:
            url = self._get_url("/clients")
            response = requests.post(url, json={"name": name}, auth=self.auth, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to create client {name}: {e}")
            return None

    def delete_client(self, client_id: str) -> bool:
        """
        Delete a client.

        Args:
            client_id: ID of the client to delete

        Returns:
            True if successful, False otherwise.
        """
        try:
            url = self._get_url(f"/clients/{client_id}")
            response = requests.delete(url, auth=self.auth, timeout=5)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to delete client {client_id}: {e}")
            return False

    def get_metrics(self) -> str:
        """
        Get Prometheus metrics.

        Returns:
            Raw metrics string.
        """
        try:
            # Metrics are usually on a separate port or root path, but based on API docs
            # they are at port 9090 by default.
            # However, the API docs also mention /api/metrics might be available or proxied.
            # For this implementation, we'll assume the main API port for simplicity
            # or allow a separate metrics URL configuration if needed.
            # Let's try the standard Prometheus port 9090 if the base URL is 8080,
            # otherwise assume it's relative.

            # Parsing base_url to switch port if needed is complex, so let's assume
            # the user might provide a separate metrics URL or we use the one from docs.
            # Docs say: "Доступно на порту 9090 (по умолчанию)."

            # For robustness, let's try to fetch from the configured base_url/metrics first
            # (if proxied), if that fails, we might need a separate config.
            # But the 'lab_ai_routing.py' uses port 9090.

            # Let's assume for this client, we stick to the API port for management.
            # If we want metrics, we might need a separate method or URL.
            # Let's try the API endpoint mentioned in docs if it exists, or fallback.
            # Docs say: "GET /metrics ... Возвращает метрики ... Доступно на порту 9090"
            # It's ambiguous if it's also on 8080/api/metrics.
            # Let's implement a safe fetch.

            # We will use a dedicated metrics URL if provided, or guess.
            metrics_url = self.base_url.replace(":8080", ":9090").replace("/api", "") + "/metrics"
            response = requests.get(metrics_url, timeout=5)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to get metrics: {e}")
            return ""
