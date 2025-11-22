# Multi-stage build for AI Routing Lab
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Create directories for models and data
RUN mkdir -p /app/models /app/data/raw /app/data/processed

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PROMETHEUS_URL=http://localhost:9090
ENV MODELS_DIR=/app/models

# Expose port for API
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "inference.predictor_service:app", "--host", "0.0.0.0", "--port", "5000"]
