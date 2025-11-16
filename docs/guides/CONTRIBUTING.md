# Contributing to AI Routing Lab

Thank you for your interest in contributing to AI Routing Lab!

---

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ai-routing-lab.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Submit a pull request

---

## Development Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

---

## Code Style

- Follow PEP 8 style guide
- Use `black` for code formatting
- Use `flake8` for linting
- Use type hints where possible

```bash
# Format code
black .

# Check linting
flake8 .

# Type checking
mypy .
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_collectors.py
```

---

## Pull Request Process

1. Update README.md if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update documentation
5. Submit PR with clear description

---

## Research Contributions

For research contributions:

1. Document methodology
2. Include experiment results
3. Provide reproducibility instructions
4. Link to related papers/publications

---

Thank you for contributing!

