# CAI Framework Integration Guide

This guide explains how to integrate CAI (Cybersecurity AI) framework with AI Routing Lab for educational and research purposes.

## Overview

CAI (Cybersecurity AI) is an open-source framework for building AI-powered security agents. This integration demonstrates how AI agents can be used to enhance ML systems for network routing optimization, security testing, and infrastructure monitoring.

## Installation

### Prerequisites

- Python 3.11+
- AI Routing Lab installed and configured
- API keys for LLM models (OpenAI, Anthropic, etc.)

### Install CAI Framework

```bash
# Install CAI framework
pip install cai-framework

# Or add to requirements-dev.txt (commented by default)
# cai-framework==0.5.5
```

### Configure Environment

Create or update `.env` file:

```bash
# LLM Model Configuration
CAI_MODEL=openai/gpt-4o
OPENAI_API_KEY=your-api-key-here

# Optional: Other model providers
ANTHROPIC_API_KEY=your-api-key-here
OLLAMA_API_BASE=http://localhost:11434
```

## Available Laboratory Works

### 1. AI-Assisted Route Optimization

**File:** `labs/lab_ai_routing.py`

**Objective:** Learn to use AI agents for automatic optimization of route selection based on ML predictions.

**Usage:**

```bash
python -m labs.lab_ai_routing
```

**Features:**
- AI agent for analyzing Prometheus metrics
- Integration with ML prediction models
- Automatic selection of optimal route
- Performance evaluation

### 2. ML System Security Testing

**File:** `labs/lab_ml_security.py`

**Objective:** Learn to use CAI framework for testing security of ML systems.

**Usage:**

```bash
python -m labs.lab_ml_security
```

**Features:**
- FastAPI endpoint security testing
- ML model security analysis
- Infrastructure security evaluation
- Automated vulnerability detection

### 3. AI Monitoring of Network Infrastructure

**File:** `labs/lab_ai_monitoring.py`

**Objective:** Learn to use AI agents for automated monitoring and analysis of network infrastructure.

**Usage:**

```bash
python -m labs.lab_ai_monitoring
```

**Features:**
- Prometheus metrics analysis
- Automated anomaly detection
- Intelligent alerting
- Performance insights

## Examples

### Basic AI Agent

```python
from cai.sdk.agents import Agent, Runner, OpenAIChatCompletionsModel

# Create agent
agent = Agent(
    name="My Agent",
    description="Description of agent",
    instructions="Agent instructions",
    model=OpenAIChatCompletionsModel(
        model="openai/gpt-4o",
        openai_client=None,
    ),
)

# Run agent
result = await Runner.run(agent, "Your query here")
```

### Route Optimization Agent

```python
from labs.lab_ai_routing import AIRoutingAgent

# Initialize agent
agent = AIRoutingAgent()

# Define route features
route_features = {
    "route_0": [25.5, 2.3, 0.95, 1.0],
    "route_1": [30.1, 3.1, 0.85, 1.2],
}

# Optimize routing
result = await agent.optimize_routing_workflow(route_features)
print(f"Best route: {result['best_route']}")
```

### Security Testing Agent

```python
from labs.lab_ml_security import MLSecurityTester

# Initialize tester
tester = MLSecurityTester(api_url="http://localhost:5000")

# Run security audit
results = await tester.run_full_security_audit()

# Check results
print(f"API Security: {results['api_security']}")
print(f"Model Security: {results['model_security']}")
```

## Integration with AI Routing Lab

### Using with ML Models

CAI agents can be integrated with existing ML models:

```python
from models.prediction.route_prediction_ensemble import RoutePredictionEnsemble
from labs.lab_ai_routing import AIRoutingAgent

# Load ML model
ml_model = RoutePredictionEnsemble(...)

# Create AI agent
agent = AIRoutingAgent()
agent.ml_ensemble = ml_model

# Use for route optimization
result = await agent.optimize_routing_workflow(route_features)
```

### Using with Prometheus

CAI agents can analyze Prometheus metrics:

```python
from labs.lab_ai_monitoring import AIMonitoringAgent

# Create monitoring agent
agent = AIMonitoringAgent(prometheus_url="http://localhost:9090")

# Monitor infrastructure
results = await agent.monitor_workflow()

# Get insights
print(results['insights'])
```

## Best Practices

1. **Error Handling:** Always check if CAI is available before use:
   ```python
   try:
       from cai.sdk.agents import Agent
       CAI_AVAILABLE = True
   except ImportError:
       CAI_AVAILABLE = False
   ```

2. **Configuration:** Use environment variables for model configuration:
   ```python
   model_name = os.getenv("CAI_MODEL", "openai/gpt-4o")
   ```

3. **Logging:** Enable logging for debugging:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

4. **Async Operations:** Use async/await for agent operations:
   ```python
   result = await Runner.run(agent, query)
   ```

## Troubleshooting

### CAI Framework Not Available

If you see "CAI framework not available":
```bash
pip install cai-framework
```

### API Key Issues

Ensure your API keys are set in `.env`:
```bash
OPENAI_API_KEY=your-key-here
```

### Model Not Found

Check available models and update `CAI_MODEL` in `.env`:
```bash
CAI_MODEL=openai/gpt-4o
# or
CAI_MODEL=anthropic/claude-3-5-sonnet
```

## Additional Resources

- CAI Framework Documentation: https://github.com/aliasrobotics/cai
- AI Routing Lab Documentation: See `docs/` directory
- Examples: See `examples/cai_integration.py`

## Contributing

When adding new laboratory works:

1. Create new file in `labs/` directory
2. Follow existing code structure
3. Add documentation
4. Update this guide
5. Add examples if applicable

## License

This integration follows the same license as AI Routing Lab (MIT License).

