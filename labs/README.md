# Laboratory Works with CAI Framework

This directory contains laboratory works demonstrating integration of CAI (Cybersecurity AI) framework with AI Routing Lab.

## Overview

CAI (Cybersecurity AI) is an open-source framework for building AI-powered security agents. These laboratory works show how AI agents can enhance ML systems for:

- Network route optimization
- Security testing of ML systems
- Infrastructure monitoring and analysis

## Prerequisites

- Python 3.11+
- AI Routing Lab installed
- CAI framework installed (optional, for running labs):
  ```bash
  pip install cai-framework
  ```
- LLM API keys configured in `.env` file

## Available Laboratory Works

### 1. AI-Assisted Route Optimization

**File:** `lab_ai_routing.py`

**Objective:** Learn to use AI agents for automatic optimization of route selection based on ML predictions of latency and jitter.

**Key Concepts:**
- AI agent creation and configuration
- Integration with ML prediction models
- Automatic route selection
- Performance evaluation

**Usage:**
```bash
python -m labs.lab_ai_routing
```

**Example:**
```python
from labs.lab_ai_routing import AIRoutingAgent

agent = AIRoutingAgent()
route_features = {
    "route_0": [25.5, 2.3, 0.95, 1.0],
    "route_1": [30.1, 3.1, 0.85, 1.2],
}
result = await agent.optimize_routing_workflow(route_features)
```

### 2. ML System Security Testing

**File:** `lab_ml_security.py`

**Objective:** Learn to use CAI framework for testing security of ML systems, including API security, model security, and infrastructure security.

**Key Concepts:**
- FastAPI endpoint security testing
- ML model security analysis (adversarial attacks, model poisoning)
- Infrastructure security evaluation
- Automated vulnerability detection

**Usage:**
```bash
python -m labs.lab_ml_security
```

**Example:**
```python
from labs.lab_ml_security import MLSecurityTester

tester = MLSecurityTester(api_url="http://localhost:5000")
results = await tester.run_full_security_audit()
```

### 3. AI Monitoring of Network Infrastructure

**File:** `lab_ai_monitoring.py`

**Objective:** Learn to use AI agents for automated monitoring and analysis of network infrastructure metrics, anomaly detection, and alerting.

**Key Concepts:**
- Prometheus metrics analysis
- Automated anomaly detection using ML models
- Intelligent alerting based on AI analysis
- Integration with existing monitoring infrastructure

**Usage:**
```bash
python -m labs.lab_ai_monitoring
```

**Example:**
```python
from labs.lab_ai_monitoring import AIMonitoringAgent

agent = AIMonitoringAgent(prometheus_url="http://localhost:9090")
results = await agent.monitor_workflow()
```

### 4. Autonomous VPN Administrator

**File:** `lab_autonomous_vpn.py`

**Objective:** Create an AI agent that autonomously manages a Masque VPN server, monitoring metrics and handling client lifecycle to ensure security and performance.

**Key Concepts:**
- Autonomous infrastructure management
- Security policy enforcement via AI
- Automated threat response (e.g., revoking compromised clients)
- Integration with VPN API

**Usage:**
```bash
python -m labs.lab_autonomous_vpn
```

**Example:**
```python
from labs.lab_autonomous_vpn import MasqueAdminAgent

admin = MasqueAdminAgent()
await admin.run_security_audit()
```

### 5. Autonomous QUIC Protocol Testing

**File:** `lab_quic_testing.py`

**Objective:** Create an AI agent that actively tests the QUIC protocol under various network conditions to identify performance bottlenecks and stability issues.

**Key Concepts:**
- Active network testing
- Automated stress testing
- Performance analysis and benchmarking
- Network profile simulation

**Usage:**
```bash
python -m labs.lab_quic_testing
```

**Example:**
```python
from labs.lab_quic_testing import QuicTestAgent

tester = QuicTestAgent()
await tester.run_stress_test()
```

### 6. Autonomous Network Orchestration

**File:** `lab_orchestrator.py`

**Objective:** Create a master AI agent (Orchestrator) that coordinates network testing and VPN management to maintain optimal system performance.

**Key Concepts:**
- Closed-loop control systems
- Multi-agent coordination (simulated via single agent with multiple tools)
- Automated remediation
- Full-stack network autonomy

**Usage:**
```bash
python -m labs.lab_orchestrator
```

**Example:**
```python
from labs.lab_orchestrator import OrchestratorAgent

orchestrator = OrchestratorAgent()
await orchestrator.run_optimization_cycle()
```

## Configuration

### Environment Variables

Create or update `.env` file:

```bash
# LLM Model Configuration
CAI_MODEL=openai/gpt-4o
OPENAI_API_KEY=your-api-key-here

# Optional: Other model providers
ANTHROPIC_API_KEY=your-api-key-here
OLLAMA_API_BASE=http://localhost:11434
```

### Prometheus Configuration

Ensure Prometheus is running and accessible:

```bash
# Default Prometheus URL
http://localhost:9090
```

### FastAPI Service

For security testing labs, ensure FastAPI service is running:

```bash
# Start inference service
python -m uvicorn inference.predictor_service:app --host 0.0.0.0 --port 5000
```

## Integration Examples

See `examples/cai_integration.py` for additional examples of CAI framework integration.

## Documentation

For detailed integration guide, see `docs/labs/CAI_INTEGRATION.md`.

## Troubleshooting

### CAI Framework Not Available

If you see "CAI framework not available":
```bash
pip install cai-framework
```

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### API Key Issues

Check your `.env` file has correct API keys:
```bash
OPENAI_API_KEY=your-key-here
```

## Contributing

When adding new laboratory works:

1. Create new file following naming convention: `lab_*.py`
2. Follow existing code structure and patterns
3. Add comprehensive docstrings
4. Update this README
5. Add examples if applicable
6. Update main documentation in `docs/labs/CAI_INTEGRATION.md`

## License

This module follows the same license as AI Routing Lab (MIT License).

