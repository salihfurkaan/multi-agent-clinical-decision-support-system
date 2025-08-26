FROM python:3.11-slim

# Prevent Python from writing .pyc files / enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml first (dependency cache)
COPY pyproject.toml ./

# Copy the actual package folder
COPY src/multiagent_clinicaldecisionsupport ./multiagent_clinicaldecisionsupport

# Install dependencies + project in editable mode
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy everything else (mcps, knowledge, tests, etc.)
COPY . .

# Run your app
CMD ["python", "-m", "multiagent_clinicaldecisionsupport.main"]
