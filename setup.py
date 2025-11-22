"""Setup script for AI Routing Lab."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-routing-lab",
    version="0.2.1",
    author="2GC CloudBridge",
    author_email="info@cloudbridge-research.ru",
    description="Predictive Route Selection using Machine Learning for Latency/Jitter Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/twogc/ai-routing-lab",
    packages=find_packages(exclude=["tests", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Networking",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-routing-collect=data.collectors.quic_test_collector:main",
            "ai-routing-train=training.train_latency_model:main",
            "ai-routing-predict=inference.predictor:main",
        ],
    },
)
