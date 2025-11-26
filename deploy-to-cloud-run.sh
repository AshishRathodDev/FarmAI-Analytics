#!/bin/bash

# ============================================================================
# FarmAI Backend - Google Cloud Run Deployment Script
# ============================================================================

set -e  # Exit on error

echo "============================================================"
echo "🌾 FarmAI Backend - Cloud Run Deployment"
echo "============================================================"

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ID="crack-decorator-468911-s1"  
REGION="asia-south1"          
SERVICE_NAME="farmai-backend"
IMAGE_NAME="farmai-backend"

# Memory and CPU configuration
MEMORY="2Gi"                  # 2GB RAM (free tier: 512Mi)
CPU="1"                       # 1 CPU
TIMEOUT="300s"                # 5 minutes timeout
MAX_INSTANCES="10"            # Max concurrent instances
MIN_INSTANCES="0"             # Min instances (0 = scale to zero)

echo ""
echo "Configuration:"
echo "  - Project ID: $PROJECT_ID"
echo "  - Region: $REGION"
echo "  - Service: $SERVICE_NAME"
echo "  - Memory: $MEMORY"
echo "  - Timeout: $TIMEOUT"
echo ""

# ============================================================================
# Step 1: Check gcloud CLI
# ============================================================================

echo "Step 1: Checking gcloud CLI..."

if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found!"
    echo "Please install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "✅ gcloud CLI found"

# ============================================================================
# Step 2: Set Project
# ============================================================================

echo ""
echo "Step 2: Setting GCP project..."

gcloud config set project $PROJECT_ID

echo "✅ Project set to: $PROJECT_ID"

# ============================================================================
# Step 3: Enable Required APIs
# ============================================================================

echo ""
echo "Step 3: Enabling required APIs..."

gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com

echo "✅ APIs enabled"

# ============================================================================
# Step 4: Build Docker Image
# ============================================================================

echo ""
echo "Step 4: Building Docker image..."
echo "This may take 5-10 minutes..."

gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME

echo "✅ Image built successfully"

# ============================================================================
# Step 5: Deploy to Cloud Run
# ============================================================================

echo ""
echo "Step 5: Deploying to Cloud Run..."

gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --memory $MEMORY \
    --cpu $CPU \
    --timeout $TIMEOUT \
    --max-instances $MAX_INSTANCES \
    --min-instances $MIN_INSTANCES \
    --allow-unauthenticated \
    --port 8080 \
    --set-env-vars "PYTHONUNBUFFERED=1,WORKERS=2"

echo "✅ Deployment complete!"

# ============================================================================
# Step 6: Get Service URL
# ============================================================================

echo ""
echo "Step 6: Getting service URL..."

SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --format 'value(status.url)')

echo ""
echo "============================================================"
echo "🎉 Deployment Successful!"
echo "============================================================"
echo ""
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test endpoints:"
echo "  - Health: $SERVICE_URL/health"
echo "  - Classes: $SERVICE_URL/api/classes"
echo "  - Predict: $SERVICE_URL/api/predict (POST)"
echo ""
echo "Update your frontend API URL to: $SERVICE_URL"
echo ""
echo "============================================================"
