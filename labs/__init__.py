"""
Laboratory works using CAI (Cybersecurity AI) Framework.

This module contains laboratory works for students demonstrating
integration of AI agents with AI Routing Lab system.

Available laboratory works:
- lab_ai_routing.py: AI-assisted route optimization
- lab_ml_security.py: ML system security testing
- lab_ai_monitoring.py: AI monitoring of network infrastructure
"""

__version__ = "0.1.0"

# Import main classes for easy access
try:
    from labs.lab_ai_routing import AIRoutingAgent
    from labs.lab_ml_security import MLSecurityTester
    from labs.lab_ai_monitoring import AIMonitoringAgent

    __all__ = [
        "AIRoutingAgent",
        "MLSecurityTester",
        "AIMonitoringAgent",
    ]
except ImportError:
    # CAI framework not available
    __all__ = []
