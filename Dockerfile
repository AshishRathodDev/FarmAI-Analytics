FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY deployment requirements
COPY requirements_deploy.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app_hf.py .

# Create temp directories
RUN mkdir -p /tmp/uploads /tmp/models /tmp/hf_cache && \
    chmod 777 /tmp/uploads /tmp/models /tmp/hf_cache

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/hf_cache

EXPOSE 8080

# Run with Gunicorn
CMD exec gunicorn app_hf:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 4 \
    --timeout 600 \
    --preload \
    --access-logfile - \
    --error-logfile -

        