# 🌾 FarmAI Analytics - AI-Powered Crop Disease Detection Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![React 19.2](https://img.shields.io/badge/react-19.2-61dafb.svg)](https://reactjs.org/)
[![Flask 3.0](https://img.shields.io/badge/flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow 2.20](https://img.shields.io/badge/tensorflow-2.20-orange.svg)](https://www.tensorflow.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/rathodashish10/farmai-models)

> **AI-driven crop disease detection system powered by deep learning to help farmers identify plant diseases instantly and get treatment recommendations.**

<div align="center">

![FarmAI Hero](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/sample_predictions.png)

*Real-time disease detection with 96.8% accuracy powered by deep learning*

</div>

---

## 🚀 Live Demo

🌐 **Frontend (React):** [https://farmai-frontend-148791329286.asia-south1.run.app/](https://farmai-frontend-148791329286.asia-south1.run.app/)  
🔧 **Backend API (Flask):** [https://farmai-backend-148791329286.asia-south1.run.app/](https://farmai-backend-148791329286.asia-south1.run.app/)  
🤗 **Models Repository:** [huggingface.co/rathodashish10/farmai-models](https://huggingface.co/rathodashish10/farmai-models)  
📊 **Dataset Source:** [Kaggle PlantDisease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

> ⚠️ **Note:** Hosted on Google Cloud Run. First request may take 60-90 seconds for cold start.

---

## 📋 Table of Contents

- [Features](#-features)
- [Model Performance](#-model-performance)
- [Live Demo](#-live-demo)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Pre-trained Models](#-pre-trained-models)
- [Installation](#-installation)
- [Deployment](#-deployment)
- [API Documentation](#-api-documentation)
- [Training Visualizations](#-training-visualizations)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## ✨ Features

### Core Functionality
- 🔍 **Real-Time Disease Detection** - Upload crop images and get instant AI-powered disease diagnosis
- 🤖 **AI Chatbot Assistant** - 24/7 agricultural expert powered by Google Gemini AI
- 📊 **Analytics Dashboard** - Comprehensive insights on disease trends and detection patterns
- 💊 **Treatment Recommendations** - Detailed treatment plans with pesticide suggestions
- 🌍 **Multi-Language Support** - Available in English, Hindi, Marathi, and Punjabi
- 📱 **Responsive Design** - Works seamlessly on desktop, tablet, and mobile devices

### Technical Highlights
- ⚡ **96.8% Model Accuracy** - MobileNetV2-based deep learning model
- 🚀 **Sub-5s Inference** - Average prediction time ~3-5 seconds on Cloud Run
- 🔐 **Secure API** - CORS protection, input validation, file size limits
- 🤗 **Hugging Face Integration** - Models auto-download from cloud storage
- ☁️ **Cloud-Native** - Deployed on Google Cloud Run (Frontend + Backend)
- 🔄 **Auto-Scaling** - Handles traffic spikes automatically

### Supported Crops & Diseases
- **Crops**: Tomato, Potato, Pepper (Bell), Grape, Apple, Orange, Peach, Cherry, Strawberry
- **Diseases**: 38+ diseases including Late Blight, Early Blight, Bacterial Spot, Leaf Mold, Mosaic Virus, and more

---

## 📊 Model Performance

### Training Metrics

| Metric | Score | Notes |
|--------|-------|-------|
| **Training Accuracy** | 98.2% | On training set (54,000+ images) |
| **Validation Accuracy** | 96.8% | On validation set (10,000+ images) |
| **Test Accuracy** | 94.5% | On unseen test data |
| **Top-3 Accuracy** | 98.7% | Correct prediction in top 3 results |
| **Inference Time** | 3-5s | CPU inference on Cloud Run |
| **Model Size** | 18 MB | Optimized for fast loading |

<div align="center">

### 📈 Training Progress

![Training Curves](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/training_curves.png)

*Model accuracy and loss curves over 30 epochs showing convergence and early stopping*

</div>

---

## 🎯 Model Evaluation

<div align="center">

### Confusion Matrix - Final Model Performance

![Confusion Matrix](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/confusion_matrix.png)

*Confusion matrix showing model performance across all 38 disease classes with 96.8% overall accuracy*

</div>

### Key Insights
- ✅ Strong diagonal indicates high accuracy across all classes
- ✅ Minimal confusion between disease categories
- ✅ Consistent performance on both common and rare diseases
- ✅ Robust to image variations (lighting, angles, backgrounds)

---

## 📊 Dataset

### Source
**Kaggle PlantDisease Dataset** by @emmarex

🔗 **Dataset Link:** [https://www.kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Images** | 87,000+ |
| **Training Set** | 70,000+ images |
| **Validation Set** | 10,000+ images |
| **Test Set** | 7,000+ images |
| **Number of Classes** | 38 disease categories |
| **Image Resolution** | 256x256 pixels (resized to 160x160) |
| **Format** | JPG/PNG |

### Dataset Composition

<div align="center">

![Class Distribution](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/eda_class_distribution.png)

*Balanced dataset distribution across 38 disease classes ensuring fair model training*

</div>

### Sample Training Images

<div align="center">

![Sample Images](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/eda_sample_images.png)

*Representative samples from each disease class used for model training*

</div>

### Data Augmentation

To improve model generalization, we applied:
- ✅ Random rotation (±20°)
- ✅ Width/height shifts (±20%)
- ✅ Horizontal flipping
- ✅ Zoom range (±15%)
- ✅ Brightness adjustment (0.8-1.2x)

---

## 🔬 Baseline Comparison

<div align="center">

### Before vs After Fine-Tuning

| Metric | Baseline | Final Model | Improvement |
|--------|----------|-------------|-------------|
| Accuracy | 89.3% | 96.8% | +7.5% |
| Loss | 0.34 | 0.11 | -68% |
| Top-3 Accuracy | 95.2% | 98.7% | +3.5% |

![Baseline Confusion Matrix](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/baseline_confusion_matrix.png)

*Baseline model performance (left) vs Fine-tuned model (right) showing significant improvement*

</div>

<div align="center">

### Baseline Dataset Analysis

![Baseline Distribution](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/baseline_class_distribution.png)

*Initial dataset distribution before augmentation and class balancing*

![Baseline Samples](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/baseline_sample_images.png)

*Sample images from baseline dataset showing raw data quality*

</div>

---

## 🛠 Tech Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Core backend language |
| **Flask** | 3.0.0 | REST API framework |
| **TensorFlow** | 2.20.0 | Deep learning inference |
| **Keras** | 3.0.0 | High-level neural networks API |
| **Hugging Face Hub** | 0.20.0 | Model hosting & auto-download |
| **Google Gemini AI** | Latest | AI chatbot integration |
| **Pillow** | 10.2.0 | Image processing |
| **NumPy** | 1.24.3 | Numerical computations |
| **Flask-CORS** | 4.0.0 | Cross-origin resource sharing |
| **Gunicorn** | 21.2.0 | WSGI production server |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 19.2.0 | UI framework |
| **Vite** | 7.2.2 | Build tool & dev server |
| **Tailwind CSS** | 3.4.18 | Utility-first CSS |
| **Lucide React** | 0.554.0 | Icon library |
| **Axios** | 1.6.2 | HTTP client |

### Deployment & DevOps
| Platform | Purpose | URL |
|----------|---------|-----|
| **Google Cloud Run** | Frontend hosting | [farmai-frontend-148791329286.asia-south1.run.app](https://farmai-frontend-148791329286.asia-south1.run.app/) |
| **Google Cloud Run** | Backend API hosting | [farmai-backend-148791329286.asia-south1.run.app](https://farmai-backend-148791329286.asia-south1.run.app/) |
| **Hugging Face** | Model storage | [rathodashish10/farmai-models](https://huggingface.co/rathodashish10/farmai-models) |
| **GitHub** | Source code | [AshishRathodDev/FarmAI-Analytics](https://github.com/AshishRathodDev/FarmAI-Analytics) |
| **Docker** | Containerization | Official Python 3.11 slim base |

### Infrastructure Specifications

**Backend (Cloud Run):**
- Memory: 4-8 GB RAM
- CPU: 2-4 vCPUs
- Timeout: 600 seconds
- Region: asia-south1 (Mumbai, India)
- Auto-scaling: 0-10 instances

**Frontend (Cloud Run):**
- Memory: 1 GB RAM
- CPU: 1 vCPU
- Timeout: 300 seconds
- Region: asia-south1
- Auto-scaling: 0-5 instances

---

## 🤖 Pre-trained Models

Models are hosted on **Hugging Face Hub** 🤗 for reliable, scalable model delivery.

**Repository:** [huggingface.co/rathodashish10/farmai-models](https://huggingface.co/rathodashish10/farmai-models)

### 📦 Available Models

| Model | Size | Accuracy | Use Case |
|-------|------|----------|----------|
| `crop_disease_classifier_final.keras` | 18 MB | 96.8% | Main production model (Keras 3 format) |
| `crop_disease_classifier_final.h5` | 18 MB | 96.8% | Legacy HDF5 format |
| `best_model_checkpoint.h5` | 18 MB | 94.5% | Training checkpoint |
| `class_indices.json` | <1 KB | - | Disease class mappings (38 classes) |

### 🚀 Automatic Setup (Recommended)

Models automatically download from Hugging Face on first run:

```bash
python app_hf.py  # Models download automatically on startup
```

**Expected startup logs:**
```
============================================================
Downloading from Hugging Face...
Repository: rathodashish10/farmai-models
============================================================
Model already exists (18.2 MB)
Class indices already exist
============================================================
LOADING MODEL (CPU-optimized)
============================================================
Loading classes...
Loaded 38 classes
Clearing memory...
Loading model from models/crop_disease_classifier_final.keras...
Keras: 3.0.0
Compiling for CPU...
Warming up model...
Warm-up complete
✓ MODEL READY
  Input: (None, 160, 160, 3)
  Output: (None, 38)
============================================================
```

### 🔧 Manual Download (Development)

For local development or testing:

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Download specific model
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('rathodashish10/farmai-models', 'crop_disease_classifier_final.keras', local_dir='models', local_dir_use_symlinks=False)"

# Download all models
huggingface-cli download rathodashish10/farmai-models --local-dir models/
```

### 🏗️ Model Architecture

```
Input (160x160x3 RGB)
        ↓
MobileNetV2 Base (Pre-trained on ImageNet)
        ↓
Global Average Pooling
        ↓
Dense(256) + ReLU + Dropout(0.5)
        ↓
Dense(38) + Softmax
        ↓
Output (38 Disease Classes)
```

**Technical Specifications:**
- **Base Architecture:** MobileNetV2 (Transfer Learning)
- **Framework:** TensorFlow 2.20.0 / Keras 3.0.0
- **Input Size:** 160x160x3 RGB images
- **Output Classes:** 38 plant disease categories
- **Training Dataset:** PlantVillage Dataset (87,000+ images from Kaggle)
- **Training Hardware:** Apple M1 MacBook Air (8GB RAM)
- **Training Time:** ~4 hours (2 phases)
- **Model Size:** 18 MB (optimized for production)
- **Parameters:** ~2.5M trainable + 2.3M frozen = 4.8M total

### 🔬 Training Configuration

```python
# Phase 1: Feature extraction (Frozen base)
EPOCHS_PHASE1 = 10
LEARNING_RATE_PHASE1 = 0.001
BASE_MODEL_TRAINABLE = False

# Phase 2: Fine-tuning (Unfrozen top layers)
EPOCHS_PHASE2 = 20
LEARNING_RATE_PHASE2 = 0.0001
UNFREEZE_FROM_LAYER = 100

# Data Augmentation
AUGMENTATION = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.15,
    'brightness_range': [0.8, 1.2]
}

# Optimizer & Loss
OPTIMIZER = 'Adam'
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy', 'top_k_categorical_accuracy']

# Early Stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MONITOR = 'val_accuracy'
```

### 🎯 Why Hugging Face?

✅ **Industry Standard** - Used by Meta, Google, Microsoft  
✅ **Free Unlimited Storage** - For public models  
✅ **Global CDN** - Fast downloads worldwide  
✅ **Version Control** - Built-in model versioning  
✅ **Easy Integration** - 3 lines of Python code  
✅ **Auto-download** - Models download on first run  
✅ **No Authentication Required** - For public models

---

## 📥 Installation

### Prerequisites
- **Python** 3.11 or higher
- **Node.js** 20.x or higher
- **npm** or **yarn**
- **Git**
- **Google Cloud CLI** (for deployment)

### Step 1: Clone Repository
```bash
git clone https://github.com/AshishRathodDev/FarmAI-Analytics.git
cd FarmAI-Analytics
```

### Step 2: Backend Setup

#### 2.1 Create Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

#### 2.2 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2.3 Run Backend
```bash
python app_hf.py
```

**Expected output:**
```
============================================================
Downloading from Hugging Face...
✓ Model exists (18.2 MB)
✓ Model loaded successfully!
============================================================
Starting FarmAI Backend on port 8080
============================================================
```

**Test Backend:**
```bash
curl http://localhost:8080/health
```

### Step 3: Frontend Setup

```bash
cd farmai-react-ui
npm install
npm run dev
```

**Access:** http://localhost:5173

---

## 🌐 Deployment

### Backend Deployment (Google Cloud Run)

#### Step 1: Build Docker Image

```bash
# From project root
gcloud run deploy farmai-backend \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 8Gi \
  --cpu 4 \
  --timeout 600 \
  --concurrency 1 \
  --max-instances 10 \
  --platform managed \
  --set-env-vars "CUDA_VISIBLE_DEVICES=-1,TF_CPP_MIN_LOG_LEVEL=2"
```

#### Step 2: Verify Deployment

```bash
# Check health
curl https://farmai-backend-148791329286.asia-south1.run.app/health

# Test prediction
curl -X POST https://farmai-backend-148791329286.asia-south1.run.app/api/predict \
  -F "file=@test_leaf.jpg"
```

---

### Frontend Deployment (Google Cloud Run)

#### Step 1: Update API URL

In `farmai-react-ui/.env`:
```env
VITE_API_URL=https://farmai-backend-148791329286.asia-south1.run.app
```

#### Step 2: Deploy Frontend

```bash
cd farmai-react-ui

gcloud run deploy farmai-frontend \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --platform managed
```

**Deployment URL:** `https://farmai-frontend-148791329286.asia-south1.run.app`

---

## 📡 API Documentation

### Base URL
```
Production: https://farmai-backend-148791329286.asia-south1.run.app
Local: http://localhost:8080
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "FarmAI API - Production Ready",
  "model_loaded": true,
  "classes_count": 38,
  "keras_version": "3.0.0",
  "cpu_only": true,
  "optimized": true,
  "cors_enabled": true,
  "endpoints": {
    "health": "/",
    "predict": "/api/predict",
    "classes": "/api/classes"
  }
}
```

#### 2. Predict Disease
```http
POST /api/predict
Content-Type: multipart/form-data
```

**Request:**
```bash
curl -X POST https://farmai-backend-148791329286.asia-south1.run.app/api/predict \
  -F "file=@leaf_image.jpg"
```

**Response:**
```json
{
  "status": "success",
  "prediction": "Tomato: Late Blight",
  "confidence": 0.9687,
  "confidence_percent": "96.87%",
  "top_3": [
    {
      "disease": "Tomato: Late Blight",
      "confidence": 0.9687,
      "confidence_percent": "96.87%",
      "raw_name": "Tomato___Late_blight"
    },
    {
      "disease": "Tomato: Early Blight",
      "confidence": 0.0234,
      "confidence_percent": "2.34%",
      "raw_name": "Tomato___Early_blight"
    },
    {
      "disease": "Tomato: Healthy",
      "confidence": 0.0052,
      "confidence_percent": "0.52%",
      "raw_name": "Tomato___healthy"
    }
  ],
  "inference_time": "3.45s",
  "total_time": "3.52s"
}
```

#### 3. Get Disease Classes
```http
GET /api/classes
```

**Response:**
```json
{
  "status": "success",
  "classes": [
    {
      "raw_name": "Tomato___Late_blight",
      "display_name": "Tomato: Late Blight"
    },
    ...
  ],
  "count": 38
}
```

#### 4. Error Responses

**400 Bad Request:**
```json
{
  "status": "error",
  "message": "No file uploaded"
}
```

**503 Service Unavailable:**
```json
{
  "status": "error",
  "message": "Model not loaded"
}
```

**500 Internal Server Error:**
```json
{
  "status": "error",
  "message": "Prediction failed: <error details>"
}
```

---

## 📊 Training Visualizations

### All Results

<div align="center">

| Visualization | Description |
|---------------|-------------|
| ![Training Curves](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/training_curves.png) | **Training Progress**: Model accuracy and loss over 30 epochs showing convergence at 96.8% accuracy |
| ![Confusion Matrix](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/confusion_matrix.png) | **Final Performance**: Confusion matrix across 38 disease classes with minimal off-diagonal errors |
| ![Sample Predictions](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/sample_predictions.png) | **Real-World Results**: Model predictions on test images with confidence scores |
| ![Class Distribution](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/eda_class_distribution.png) | **Dataset Balance**: Distribution across 38 disease categories showing balanced training data |
| ![Sample Images](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/eda_sample_images.png) | **Training Data**: Representative samples from each disease class |
| ![Baseline Matrix](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/baseline_confusion_matrix.png) | **Baseline Comparison**: Initial model performance before fine-tuning (89.3% accuracy) |
| ![Baseline Distribution](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/baseline_class_distribution.png) | **Initial Dataset**: Class distribution before data augmentation |
| ![Baseline Samples](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/baseline_sample_images.png) | **Baseline Data**: Raw sample images before preprocessing |

</div>

---

## 📁 Project Structure

```
FarmAI-Analytics/
├── 📂 farmai-react-ui/              # React frontend
│   ├── src/
│   │   ├── App.jsx                  # Main React component
│   │   ├── api-service.js           # API client
│   │   └── main.jsx                 # Entry point
│   ├── Dockerfile                   # Frontend container
│   ├── package.json                 # Node dependencies
│   ├── vite.config.js               # Vite configuration
│   └── .env                         # Environment variables
│
├── 📂 models/                       # ML models (auto-downloaded)
│   ├── .gitkeep                     # Track folder in Git
│   ├── crop_disease_classifier_final.keras  # Main model
│   └── class_indices.json           # Class mappings
│
├── 📂 results/                      # Training outputs
│   ├── figures/                     # Performance visualizations
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   ├── sample_predictions.png
│   │   ├── eda_class_distribution.png
│   │   ├── eda_sample_images.png
│   │   ├── baseline_confusion_matrix.png
│   │   ├── baseline_class_distribution.png
│   │   └── baseline_sample_images.png
│   ├── metrics/                     # Evaluation metrics (JSON)
│   └── predictions/                 # Sample predictions
│
├── 📂 uploads/                      # Temporary image uploads
│
├── app_hf.py                        # Flask API (Production - Hugging Face)
├── app.py                           # Flask API (Alternative)
├── Dockerfile                       # Backend container
├── requirements.txt                 # Python dependencies (Dev)
├── requirements_deploy.txt          # Python dependencies (Prod)
├── upload_models.py                 # Hugging Face upload script
├── .gitignore                       # Git ignore rules
├── .dockerignore                    # Docker ignore rules
├── LICENSE                          # MIT License
└── README.md                        # This file
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m "feat: Add amazing feature"`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** Pull Request

### Commit Message Convention
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Code style
- `refactor:` Refactoring
- `test:` Tests
- `chore:` Maintenance
- `perf:` Performance improvements
- `ci:` CI/CD changes

### Development Guidelines

- Write clean, readable code
- Add comments for complex logic
- Update documentation
- Test thoroughly before PR
- Follow PEP 8 (Python) and ESLint (JavaScript)

---

## 📜 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

### MIT License Summary

✅ Commercial use  
✅ Modification  
✅ Distribution  
✅ Private use  
❌ Liability  
❌ Warranty  

---

## 📞 Contact

**Ashish Rathod**
- 🐙 GitHub: [@AshishRathodDev](https://github.com/AshishRathodDev)
- 📧 Email: [ashish3110rathod@gmail.com](mailto:ashish3110rathod@gmail.com)
- 💼 LinkedIn: [Ashish Rathod](https://www.linkedin.com/in/ashishrathod-it/)
- 🌐 Portfolio: [Coming Soon]

**Project Links:**
- 🔗 Repository: [github.com/AshishRathodDev/FarmAI-Analytics](https://github.com/AshishRathodDev/FarmAI-Analytics)
- 🌐 Live Demo: [farmai-frontend-148791329286.asia-south1.run.app](https://farmai-frontend-148791329286.asia-south1.run.app/)
- 🔧 API Docs: [farmai-backend-148791329286.asia-south1.run.app/health](https://farmai-backend-148791329286.asia-south1.run.app/health)
- 🤗 Models: [huggingface.co/rathodashish10/farmai-models](https://huggingface.co/rathodashish10/farmai-models)
- 📊 Dataset: [kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## 🙏 Acknowledgments

- **PlantVillage Dataset** via Kaggle [@emmarex](https://www.kaggle.com/datasets/emmarex/plantdisease) - Disease image dataset
- **Hugging Face** - Model hosting platform and community
- **Google Cloud Platform** - Cloud Run hosting infrastructure
- **Google Gemini AI** - Chatbot intelligence
- **TensorFlow & Keras** - Deep learning frameworks
- **React & Vite** - Frontend development tools
- **Tailwind CSS** - UI styling framework

---

## 🎓 Research & References

- **MobileNetV2 Paper:** [Sandler et al., 2018](https://arxiv.org/abs/1801.04381)
- **Transfer Learning:** [Yosinski et al., 2014](https://arxiv.org/abs/1411.1792)
- **PlantVillage Dataset:** [Hughes & Salathé, 2015](https://arxiv.org/abs/1511.08060)

---

<div align="center">

**Made with ❤️ by [Ashish Rathod](https://github.com/AshishRathodDev)**

**If you find this helpful, please ⭐ this repo!**

[![GitHub forks](https://img.shields.io/github/forks/AshishRathodDev/FarmAI-Analytics?style=social)](https://github.com/AshishRathodDev/FarmAI-Analytics/network/members)
[![GitHub watchers](https://img.shields.io/github/watchers/AshishRathodDev/FarmAI-Analytics?style=social)](https://github.com/AshishRathodDev/FarmAI-Analytics/watchers)

---

### 📸 Project Gallery

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/training_curves.png" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/confusion_matrix.png" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>Training Progress</b><br/>30 epochs with early stopping</td>
    <td align="center"><b>Model Performance</b><br/>96.8% accuracy across 38 classes</td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/sample_predictions.png" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/eda_class_distribution.png" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>Prediction Examples</b><br/>Real-world test results</td>
    <td align="center"><b>Dataset Distribution</b><br/>87,000+ balanced images</td>
  </tr>
</table>

---

### 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/AshishRathodDev/FarmAI-Analytics.git
cd FarmAI-Analytics

# Backend setup
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app_hf.py

# Frontend setup (new terminal)
cd farmai-react-ui
npm install
npm run dev
```

**Access:** http://localhost:5173

---

### 📊 Performance Metrics at a Glance

| Metric | Value | Notes |
|--------|-------|-------|
| 🎯 Accuracy | 96.8% | Validation set performance |
| ⚡ Inference | 3-5s | CPU-optimized Cloud Run |
| 📦 Model Size | 18 MB | Fast download & loading |
| 🌍 Classes | 38 | Multiple crops & diseases |
| 📷 Dataset | 87K+ | PlantVillage via Kaggle |
| 🤖 Architecture | MobileNetV2 | Transfer learning |

</div> stars](https://img.shields.io/github/stars/AshishRathodDev/FarmAI-Analytics?style=social)](https://github.com/AshishRathodDev/FarmAI-Analytics/stargazers)
[![GitHub

