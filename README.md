# 🌾 FarmAI Analytics - AI-Powered Crop Disease Detection Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![React 19.2](https://img.shields.io/badge/react-19.2-61dafb.svg)](https://reactjs.org/)
[![Flask 3.0](https://img.shields.io/badge/flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow 2.20](https://img.shields.io/badge/tensorflow-2.20-orange.svg)](https://www.tensorflow.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/rathodashish10/farmai-models)

> **AI-driven crop disease detection system powered by deep learning to help farmers identify plant diseases instantly and get treatment recommendations.**

<div align="center">

![Sample Predictions](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/sample_predictions.png)

</div>

---

## 🚀 Live Demo

🌐 **Production App:** [https://farm-ai-ten.vercel.app/](https://farm-ai-ten.vercel.app/)  
🔧 **Backend API:** [https://farmai-analytics.onrender.com](https://farmai-analytics.onrender.com)  
🤗 **Models Repository:** [huggingface.co/rathodashish10/farmai-models](https://huggingface.co/rathodashish10/farmai-models)

> ⚠️ **Note:** Free tier services may take 30-50 seconds to wake up on first request.

---

## 📋 Table of Contents

- [Features](#-features)
- [Model Performance](#-model-performance)
- [Live Demo](#-live-demo)
- [Tech Stack](#-tech-stack)
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
- 🚀 **Sub-200ms Inference** - Average prediction time ~200ms
- 🔐 **Secure API** - CORS protection, input validation, file size limits
- 🤗 **Hugging Face Integration** - Models auto-download from cloud storage
- 📈 **Production-Ready** - Deployed on Vercel (Frontend) + Render (Backend)

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
| **Inference Time** | ~200ms | CPU inference on Render free tier |
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

## 📸 Dataset Visualization

<div align="center">

### Class Distribution Analysis

![Class Distribution](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/eda_class_distribution.png)

*Balanced dataset distribution across 38 disease classes ensuring fair model training*

### Sample Training Images

![Sample Images](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/eda_sample_images.png)

*Representative samples from each disease class used for model training*

</div>

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

---

## 🛠 Tech Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Core backend language |
| **Flask** | 3.0.0 | REST API framework |
| **TensorFlow** | 2.20.0 | Deep learning inference |
| **Hugging Face Hub** | 0.20.0 | Model hosting & auto-download |
| **Google Gemini AI** | Latest | AI chatbot integration |
| **Pillow** | 10.2.0 | Image processing |
| **NumPy** | 1.24.3 | Numerical computations |
| **Gunicorn** | 21.2.0 | WSGI production server |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 19.2.0 | UI framework |
| **Vite** | 7.2.2 | Build tool & dev server |
| **Tailwind CSS** | 3.4.18 | Utility-first CSS |
| **Lucide React** | 0.554.0 | Icon library |

### Deployment & DevOps
| Platform | Purpose | URL |
|----------|---------|-----|
| **Vercel** | Frontend hosting | [farm-ai-ten.vercel.app](https://farm-ai-ten.vercel.app/) |
| **Render** | Backend API hosting | [farmai-analytics.onrender.com](https://farmai-analytics.onrender.com) |
| **Hugging Face** | Model storage | [rathodashish10/farmai-models](https://huggingface.co/rathodashish10/farmai-models) |
| **GitHub** | Source code | [AshishRathodDev/FarmAI-Analytics](https://github.com/AshishRathodDev/FarmAI-Analytics) |

---

## 🤖 Pre-trained Models

Models are hosted on **Hugging Face Hub** 🤗 for reliable, scalable model delivery.

**Repository:** [huggingface.co/rathodashish10/farmai-models](https://huggingface.co/rathodashish10/farmai-models)

### 📦 Available Models

| Model | Size | Accuracy | Use Case |
|-------|------|----------|----------|
| `crop_disease_classifier_final.h5` | 18 MB | 96.8% | Main CNN classifier (Production) |
| `best_crop_disease_model.h5` | 20 MB | 95.2% | Alternative checkpoint |
| `best_model_checkpoint.h5` | 18 MB | 94.5% | Training checkpoint |
| `class_indices.json` | <1 KB | - | Disease class mappings |

### 🚀 Automatic Setup (Recommended)

Models automatically download from Hugging Face on first run:

```bash
python app_hf.py  # Models download automatically on startup
```

**Expected startup logs:**
```
📥 Downloading crop_disease_classifier_final.h5 from Hugging Face...
✅ Model downloaded successfully!
📥 Downloading class_indices.json from Hugging Face...
✅ Class indices downloaded successfully!
✅ Model loaded successfully!
✅ Loaded 38 disease classes
```

### 🔧 Manual Download (Development)

For local development or testing:

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Download specific model
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('rathodashish10/farmai-models', 'crop_disease_classifier_final.h5', local_dir='models', local_dir_use_symlinks=False)"

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
Dense(256) + ReLU
        ↓
Dropout(0.5)
        ↓
Dense(38) + Softmax
        ↓
Output (38 Disease Classes)
```

**Technical Specifications:**
- **Base Architecture:** MobileNetV2 (Transfer Learning)
- **Framework:** TensorFlow 2.20.0 / Keras
- **Input Size:** 160x160x3 RGB images
- **Output Classes:** 38 plant disease categories
- **Training Dataset:** PlantVillage Dataset (54,000+ images)
- **Training Hardware:** M1 MacBook Air (Apple Silicon)
- **Training Time:** ~4 hours
- **Model Size:** 18 MB (optimized for production)

### 🔬 Training Configuration

```python
# Phase 1: Feature extraction (Frozen base)
EPOCHS_PHASE1 = 10
LEARNING_RATE_PHASE1 = 0.001
BASE_MODEL_TRAINABLE = False

# Phase 2: Fine-tuning (Unfrozen top layers)
EPOCHS_PHASE2 = 5
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
```

### 🎯 Why Hugging Face?

✅ **Industry Standard** - Used by Meta, Google, Microsoft  
✅ **Free Unlimited Storage** - For public models  
✅ **Global CDN** - Fast downloads worldwide  
✅ **Version Control** - Built-in model versioning  
✅ **Easy Integration** - 3 lines of Python code  
✅ **Auto-download** - Models download on first run  

---

## 📥 Installation

### Prerequisites
- **Python** 3.11 or higher
- **Node.js** 20.x or higher
- **npm** or **yarn**
- **Git**

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
python app.py
```

**Expected output:**
```
📥 Downloading crop_disease_classifier_final.h5...
✅ Model downloaded!
✅ Model loaded successfully!
🚀 Starting server on http://0.0.0.0:5050
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

### Backend Deployment (Render)

#### Step 1: Prepare Repository

1. **Ensure files are in place:**
   - `app.py` - Main Flask application
   - `requirements_deploy.txt` - Production dependencies
   - `models/.gitkeep` - Track empty models folder

2. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "deploy: prepare for Render deployment"
   git push origin main
   ```

#### Step 2: Create Render Web Service

1. **Go to:** [https://dashboard.render.com](https://dashboard.render.com)
2. **Click:** "New" → "Web Service"
3. **Connect GitHub:** Select `FarmAI-Analytics` repository
4. **Configure:**

   | Setting | Value |
   |---------|-------|
   | **Name** | `farmai-analytics` |
   | **Region** | Singapore (closest to India) |
   | **Branch** | `main` |
   | **Runtime** | Python 3 |
   | **Build Command** | `pip install -r requirements_deploy.txt` |
   | **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT` |
   | **Instance Type** | Free |

5. **Environment Variables** (Optional):
   ```
   PYTHON_VERSION = 3.11.0
   PORT = 5050
   ```

6. **Click:** "Create Web Service"

#### Step 3: Wait for Deployment

**Build logs should show:**
```
==> Building...
==> Installing Python 3.11...
==> pip install -r requirements_deploy.txt
Successfully installed flask tensorflow huggingface_hub...
==> Starting server...
📥 Downloading models from Hugging Face...
✅ Model downloaded!
🚀 Server started on port 5050
```

**Deployment URL:** `https://farmai-analytics.onrender.com`

---

### Frontend Deployment (Vercel)

#### Step 1: Prepare Frontend

1. **Update API URL in `.env`:**
   ```env
   VITE_API_URL=https://farmai-analytics.onrender.com/api
   ```

2. **Build locally to test:**
   ```bash
   cd farmai-react-ui
   npm run build
   npm run preview
   ```

#### Step 2: Deploy to Vercel

**Option A: Vercel CLI**
```bash
npm install -g vercel
cd farmai-react-ui
vercel --prod
```

**Option B: Vercel Dashboard**
1. Go to: [https://vercel.com/new](https://vercel.com/new)
2. Import: `AshishRathodDev/FarmAI-Analytics`
3. **Root Directory:** `farmai-react-ui`
4. **Framework Preset:** Vite
5. **Environment Variables:**
   ```
   VITE_API_URL = https://farmai-analytics.onrender.com/api
   ```
6. Click "Deploy"

**Deployment URL:** `https://farm-ai-ten.vercel.app`

---

## 📡 API Documentation

### Base URL
```
Production: https://farmai-analytics.onrender.com/api
Local: http://localhost:5050/api
```

### Endpoints

#### 1. Health Check
```http
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "message": "FarmAI API is running!",
  "model_loaded": true,
  "classes_loaded": 38,
  "huggingface_repo": "rathodashish10/farmai-models"
}
```

#### 2. Predict Disease
```http
POST /api/predict
Content-Type: multipart/form-data
```

**Request:**
```bash
curl -X POST https://farmai-analytics.onrender.com/api/predict \
  -F "file=@leaf_image.jpg"
```

**Response:**
```json
{
  "status": "success",
  "prediction": "Tomato Late Blight",
  "confidence": 0.942,
  "top_3": [
    {"disease": "Tomato Late Blight", "confidence": 0.942},
    {"disease": "Tomato Early Blight", "confidence": 0.034},
    {"disease": "Tomato Healthy", "confidence": 0.012}
  ]
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
    "Tomato Late Blight",
    "Tomato Early Blight",
    ...
  ],
  "count": 38
}
```

---

## 📊 Training Visualizations

### All Results

<div align="center">

| Visualization | Description |
|---------------|-------------|
| ![Training Curves](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/training_curves.png) | **Training Progress**: Model accuracy and loss over 30 epochs |
| ![Confusion Matrix](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/confusion_matrix.png) | **Final Performance**: 96.8% accuracy across 38 classes |
| ![Sample Predictions](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/sample_predictions.png) | **Real-World Results**: Model predictions on test images |
| ![Class Distribution](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/eda_class_distribution.png) | **Dataset Balance**: Distribution across disease categories |
| ![Baseline Matrix](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/baseline_confusion_matrix.png) | **Baseline Comparison**: Pre-tuning performance |
| ![Baseline Distribution](https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/baseline_class_distribution.png) | **Initial Dataset**: Class distribution before augmentation |

</div>

---

## 📁 Project Structure

```
FarmAI-Analytics/
├── 📂 farmai-react-ui/          # React frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api-service.js
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── .env
│
├── 📂 models/                   # ML models (auto-downloaded)
│   ├── .gitkeep                 # Track folder in Git
│   └── (models download here)
│
├── 📂 results/                  # Training outputs
│   ├── figures/                 # Performance visualizations
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   ├── sample_predictions.png
│   │   ├── eda_class_distribution.png
│   │   ├── baseline_confusion_matrix.png
│   │   └── baseline_class_distribution.png
│   ├── metrics/                 # Evaluation metrics (JSON)
│   └── predictions/             # Sample predictions
│
├── 📂 uploads/                  # Temporary image uploads
│
├── app.py                       # Flask API (Production)
├── requirements.txt             # Python dependencies (Dev)
├── requirements_deploy.txt      # Python dependencies (Prod)
├── upload_models.py             # Hugging Face upload script
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
└── README.md                    # This file
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

---

## 📜 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file.

---

## 📞 Contact

**Ashish Rathod**
- 🐙 GitHub: [@AshishRathodDev](https://github.com/AshishRathodDev)
- 📧 Email: ashish3110rathod@gmail.com
- 💼 LinkedIn: [Ashish Rathod](https://www.linkedin.com/in/ashishrathod-it/)

**Project Links:**
- 🔗 Repository: [github.com/AshishRathodDev/FarmAI-Analytics](https://github.com/AshishRathodDev/FarmAI-Analytics)
- 🌐 Live Demo: [farm-ai-ten.vercel.app](https://farm-ai-ten.vercel.app/)
- 🤗 Models: [huggingface.co/rathodashish10/farmai-models](https://huggingface.co/rathodashish10/farmai-models)

---

## 🙏 Acknowledgments

- **PlantVillage Dataset** - Disease image dataset
- **Hugging Face** - Model hosting platform
- **Google Gemini AI** - Chatbot intelligence
- **TensorFlow** - Deep learning framework
- **Vercel** - Frontend hosting
- **Render** - Backend hosting

---

<div align="center">

**Made with ❤️ by [Ashish Rathod](https://github.com/AshishRathodDev)**

**If you find this helpful, please ⭐ this repo!**

[![GitHub stars](https://img.shields.io/github/stars/AshishRathodDev/FarmAI-Analytics?style=social)](https://github.com/AshishRathodDev/FarmAI-Analytics/stargazers)

---

### 📸 Project Gallery

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/training_curves.png" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/confusion_matrix.png" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>Training Progress</b></td>
    <td align="center"><b>Model Performance</b></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/sample_predictions.png" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/AshishRathodDev/FarmAI-Analytics/main/results/figures/eda_class_distribution.png" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>Prediction Examples</b></td>
    <td align="center"><b>Dataset Distribution</b></td>
  </tr>
</table>

</div>

