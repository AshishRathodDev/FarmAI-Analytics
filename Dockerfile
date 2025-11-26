# ============================================================================
# FarmAI Backend - Google Cloud Run Dockerfile (FIXED)
# Works with TensorFlow, Keras 3, HuggingFace, PIL, gunicorn, etc.
# ============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (your deployment requirements, can rename as needed)
COPY requirements_deploy.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and (if present) models folder
COPY app_hf.py .
COPY models/ ./models/

# Create directories (if needed)
RUN mkdir -p uploads models logs

# Expose port
ENV PORT=8080
EXPOSE 8080

# Run using gunicorn - workers usually 1-2 for ML on <2GB RAM
CMD exec gunicorn app_hf:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 2 \
    --timeout 300 \
    --preload
