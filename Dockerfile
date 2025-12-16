# Use Python 3.11 slim for a lightweight base image
FROM python:3.11-slim

# Environment variables to prevent Python from generating .pyc files
# and force unbuffered output. Also disable pip cache for smaller layers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies (e.g., gcc for some ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy dependency list first to leverage Docker layer caching
COPY requirements/prod.txt requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the entire project into the container
COPY . /app
WORKDIR /app

ENV PYTHONPATH=/app
ENV MLPROJECT_CONFIG_PATH=/app/mlproject/configs/experiments/etth1.yaml

# Install the local package so `import mlproject` works everywhere
RUN pip install .

# Expose port for the API (FastAPI defaults to 8000)
EXPOSE 8000

# Create a non-root user for better security (recommended for Kubernetes)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Default command: start the API server
# (Training jobs will override this command in Kubernetes Jobs)
CMD ["uvicorn", "mlproject.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
