import os
from typing import Optional

from cai.sdk.agents import function_tool
from labs.tools.masque_client import MasqueVPNClient

# Initialize client with env vars or defaults
VPN_API_URL = os.getenv("MASQUE_VPN_API_URL", "http://localhost:8080")
VPN_AUTH_USER = os.getenv("MASQUE_VPN_USER", "admin")
VPN_AUTH_PASS = os.getenv("MASQUE_VPN_PASS", "admin")

client = MasqueVPNClient(base_url=VPN_API_URL, auth=(VPN_AUTH_USER, VPN_AUTH_PASS))

@function_tool
def list_vpn_clients() -> str:
    """
    List all registered VPN clients.
    
    Returns:
        JSON string with list of clients.
    """
    clients = client.list_clients()
    return str(clients)

@function_tool
def create_vpn_client(name: str) -> str:
    """
    Create a new VPN client.
    
    Args:
        name: Name of the client to create.
        
    Returns:
        JSON string with created client details or error message.
    """
    result = client.create_client(name)
    if result:
        return f"Successfully created client: {result}"
    return "Failed to create client."

@function_tool
def delete_vpn_client(client_id: str) -> str:
    """
    Delete (revoke) a VPN client.
    
    Args:
        client_id: ID of the client to delete.
        
    Returns:
        Success or error message.
    """
    if client.delete_client(client_id):
        return f"Successfully deleted client {client_id}."
    return f"Failed to delete client {client_id}."

@function_tool
def get_vpn_metrics() -> str:
    """
    Get current VPN server metrics (Prometheus format).
    
    Returns:
        String containing Prometheus metrics.
    """
    metrics = client.get_metrics()
    if metrics:
        # Truncate if too long to avoid token limit issues, or return summary
        # For now return first 2000 chars as a sample
        return metrics[:2000] + "\n... (truncated)" if len(metrics) > 2000 else metrics
    return "Failed to retrieve metrics."
