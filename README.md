# 🌾 FarmAI Analytics - AI-Powered Crop Disease Detection Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React 19.2](https://img.shields.io/badge/react-19.2-61dafb.svg)](https://reactjs.org/)
[![Flask 3.0](https://img.shields.io/badge/flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow 2.15](https://img.shields.io/badge/tensorflow-2.15-orange.svg)](https://www.tensorflow.org/)

> **AI-driven crop disease detection system powered by deep learning to help farmers identify plant diseases instantly and get treatment recommendations.**

![FarmAI Banner](https://via.placeholder.com/1200x300/10b981/ffffff?text=FarmAI+Analytics+-+Crop+Disease+Detection)

---

## 📋 Table of Contents

- [Features](#-features)
- [Live Demo](#-live-demo)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Model Information](#-model-information)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)

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
- ⚡ **95.3% Model Accuracy** - EfficientNetB0-based deep learning model
- 🚀 **Sub-second Inference** - Average prediction time ~80ms
- 🔐 **Secure API** - Rate limiting, CORS protection, input validation
- 💾 **SQLite Database** - Comprehensive analytics and user tracking
- 📈 **Real-Time Metrics** - Track prediction confidence, response times, and usage patterns

### Supported Crops & Diseases
- **Crops**: Tomato, Potato, Pepper (Bell), Grape, Apple, Orange, Peach, Cherry, Strawberry
- **Diseases**: 38+ diseases including Late Blight, Early Blight, Bacterial Spot, Leaf Mold, Mosaic Virus, and more

---

## 🎯 Live Demo

<<<<<<< HEAD
🌐 **Frontend**: [http://localhost:5173](http://localhost:5173)  
🔧 **Backend API**: [http://localhost:5050](http://localhost:5050)
=======
🌐 **FarmAI Production Demo:** [https://farm-ai-ten.vercel.app/](https://farm-ai-ten.vercel.app/)
>>>>>>> 5212706977435d4bb40a86ed872cf977d3ae1675

### Quick Demo Credentials
```
Farmer ID: demo_farmer_001
Location: Pune, Maharashtra, India
```

---

## 🛠 Tech Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Core backend language |
| **Flask** | 3.0.0 | REST API framework |
| **TensorFlow** | 2.15.0 | Deep learning model inference |
| **Google Gemini AI** | 0.3.2 | AI chatbot integration |
| **SQLite** | 3.x | Database for analytics |
| **Pillow** | 10.2.0 | Image processing |
| **NumPy** | 1.24.3 | Numerical computations |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 19.2.0 | UI framework |
| **Vite** | 7.2.2 | Build tool & dev server |
| **Tailwind CSS** | 3.4.18 | Utility-first CSS framework |
| **Lucide React** | 0.554.0 | Icon library |

### AI/ML
- **Model Architecture**: EfficientNetB0 (Transfer Learning)
- **Training Dataset**: PlantVillage Dataset
- **Input Shape**: 160x160x3 RGB images
- **Output Classes**: 5 primary disease categories (expandable to 38+)
- **Training Framework**: TensorFlow/Keras

---

## 🏗 System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Client Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Web App    │  │  Mobile App  │  │   REST API   │      │
│  │ (React/Vite) │  │  (Future)    │  │   Clients    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
          ┌──────────────────▼─────────────────────┐
          │        API Gateway (Flask)              │
          │  ┌──────────┐  ┌──────────┐           │
          │  │   CORS   │  │   Rate   │           │
          │  │  Handler │  │  Limiter │           │
          │  └──────────┘  └──────────┘           │
          └──────────────────┬─────────────────────┘
                             │
          ┌──────────────────┴─────────────────────┐
          │         Application Layer               │
          │  ┌─────────┐ ┌──────────┐ ┌─────────┐ │
          │  │Prediction│ │ Chatbot  │ │Analytics│ │
          │  │ Service  │ │ Service  │ │ Engine  │ │
          │  └────┬────┘ └────┬─────┘ └────┬────┘ │
          └───────┼───────────┼────────────┼───────┘
                  │           │            │
          ┌───────▼───────────▼────────────▼───────┐
          │         Data Layer                      │
          │  ┌─────────┐ ┌──────────┐ ┌─────────┐ │
          │  │   ML    │ │  SQLite  │ │ Google  │ │
          │  │  Model  │ │ Database │ │ Gemini  │ │
          │  └─────────┘ └──────────┘ └─────────┘ │
          └─────────────────────────────────────────┘
```

---

## 📥 Installation

### Prerequisites
- **Python** 3.8 or higher
- **Node.js** 20.x or higher
- **npm** or **yarn**
- **Git**
- **Google Gemini API Key** (for chatbot functionality)

### Step 1: Clone Repository
```bash
git clone https://github.com/AshishRathodDev/FarmAI-Analytics.git
cd FarmAI-Analytics
```

### Step 2: Backend Setup

#### 2.1 Create Virtual Environment
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

#### 2.2 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2.3 Configure Environment Variables
```bash
# Create .env file
cp .env.example .env

# Edit .env and add your Google API Key
nano .env
```

**`.env` file content:**
```bash
GOOGLE_API_KEY=your_google_gemini_api_key_here
DATABASE_PATH=farmer_analytics.db
MODEL_PATH=models/crop_disease_model.h5
```

> 🔑 **Get Google Gemini API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)

#### 2.4 Initialize Database
```bash
python -c "from src.database_manager import FarmAIDatabaseManager; db = FarmAIDatabaseManager(); print('✅ Database initialized')"
```

#### 2.5 Download/Place Model File
```bash
# Ensure your trained model is in models/ directory
# Expected file: models/crop_disease_classifier_final.h5
# If missing, you need to train the model first (see Model Training section)
```

### Step 3: Frontend Setup

#### 3.1 Navigate to Frontend Directory
```bash
cd farmai-react-ui
```

#### 3.2 Install Dependencies
```bash
npm install
# or
yarn install
```

#### 3.3 Configure Environment
```bash
# Create .env file
cp .env.example .env
```

**`farmai-react-ui/.env` content:**
```bash
VITE_API_URL=http://localhost:5050/api
```

---

## 🚀 Usage

### Running the Application

#### Terminal 1: Start Backend API
```bash
# From project root
source venv/bin/activate  # On Windows: venv\Scripts\activate
python api.py
```

**Expected output:**
```
==================================================
 FarmAI Backend Service Starting...
==================================================
✅ All resources initialized!
API will be serving on host 0.0.0.0 port 5050
 * Running on http://127.0.0.1:5050
```

#### Terminal 2: Start Frontend
```bash
# From farmai-react-ui directory
npm run dev
# or
yarn dev
```

**Expected output:**
```
VITE v7.2.2  ready in 1234 ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

### Access the Application
1. Open browser and navigate to: **http://localhost:5173**
2. Default landing page: **Dashboard**
3. Start by uploading a crop image in the **Scanner** page

---

## ⚙️ Configuration

### Backend Configuration (`src/config.py`)
```python
# Model Configuration
MODEL_ARCHITECTURE = "EfficientNetB0"
IMG_SIZE = (160, 160)
BATCH_SIZE = 4
CONFIDENCE_THRESHOLD = 0.65

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 5050
CORS_ORIGINS = ["http://localhost:5173"]

# Database Configuration
DATABASE_PATH = "farmer_analytics.db"
```

### Frontend Configuration (`farmai-react-ui/vite.config.js`)
```javascript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5050',
        changeOrigin: true,
      }
    }
  }
})
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:5050/api
```

### Endpoints

#### 1. Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "online",
  "message": "FarmAI API is running",
  "model_loaded": true,
  "timestamp": "2025-01-20T10:30:00Z"
}
```

---

#### 2. Predict Disease
```http
POST /api/predict
Content-Type: multipart/form-data
```

**Request:**
```bash
curl -X POST http://localhost:5050/api/predict \
  -F "file=@/path/to/leaf_image.jpg"
```

**Response:**
```json
{
  "status": "success",
  "prediction": "Tomato Late Blight",
  "confidence": 0.942,
  "confidence_percentage": 94.2,
  "severity": "High",
  "treatment": "Apply Mancozeb fungicide (2g/L) within 24 hours...",
  "top_3": [
    {"disease": "Tomato Late Blight", "confidence": 0.942},
    {"disease": "Tomato Early Blight", "confidence": 0.034},
    {"disease": "Tomato Healthy", "confidence": 0.012}
  ],
  "prediction_time": 0.082
}
```

---

#### 3. Chat with AI Assistant
```http
POST /api/chat
Content-Type: application/json
```

**Request:**
```json
{
  "message": "How do I treat tomato late blight?",
  "language": "English"
}
```

**Response:**
```json
{
  "success": true,
  "response": "For Tomato Late Blight treatment:\n1. Remove infected leaves immediately\n2. Apply Mancozeb fungicide at 2g/L...",
  "timestamp": "2025-01-20T10:35:00Z"
}
```

---

#### 4. Get Analytics
```http
GET /api/analytics
```

**Response:**
```json
{
  "total_queries": 1247,
  "total_farmers": 234,
  "avg_response_time": 0.85,
  "model_accuracy": 95.3,
  "top_disease": "Tomato Late Blight",
  "total_predictions": 1580
}
```

---

#### 5. Get Disease Distribution
```http
GET /api/analytics/diseases?limit=10
```

**Response:**
```json
{
  "diseases": [
    {
      "disease_name": "Tomato Late Blight",
      "count": 234,
      "avg_confidence": 0.912,
      "severity": "High"
    }
  ]
}
```

---

## 📁 Project Structure
```
FarmAI-Analytics/
├── 📂 farmai-react-ui/          # React frontend application
│   ├── 📂 src/
│   │   ├── App.jsx              # Main React component
│   │   ├── api-service.js       # API integration layer
│   │   ├── main.jsx             # Entry point
│   │   └── index.css            # Global styles
│   ├── 📂 public/               # Static assets
│   ├── package.json             # Frontend dependencies
│   ├── vite.config.js           # Vite configuration
│   └── tailwind.config.js       # Tailwind CSS config
│
├── 📂 src/                      # Backend source code
│   ├── config.py                # Configuration settings
│   ├── crop_classifier.py       # ML model inference
│   ├── chatbot_agent.py         # Google Gemini integration
│   ├── database_manager.py      # SQLite operations
│   ├── analytics_engine.py      # Analytics processing
│   ├── data_utils.py            # Data preprocessing
│   ├── model.py                 # Model architecture
│   ├── train.py                 # Training pipeline
│   └── evaluate.py              # Model evaluation
│
├── 📂 models/                   # ML models and configs
│   ├── crop_disease_classifier_final.h5  # Trained model
│   └── class_indices.json       # Class mappings
│
├── 📂 scripts/                  # Utility scripts
│   ├── run_training.py          # Model training script
│   ├── run_evaluation.py        # Model evaluation
│   └── setup_data.py            # Data preparation
│
├── 📂 logs/                     # Application logs
├── 📂 data/                     # Dataset directory
│   ├── raw/                     # Raw images
│   └── processed/               # Split data
│
├── api.py                       # Flask API entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
└── README.md                    # This file
```

---

## 🧠 Model Information

### Model Details
- **Architecture**: EfficientNetB0 (Transfer Learning)
- **Base Model**: Pre-trained on ImageNet
- **Framework**: TensorFlow 2.15 / Keras
- **Input Size**: 160x160x3 (RGB images)
- **Output**: 5 disease classes (expandable to 38+)
- **Training Dataset**: PlantVillage Dataset (~54,000 images)
- **Accuracy**: 95.3% on test set
- **Inference Time**: ~80ms per image (CPU)

### Training Configuration
```python
# Phase 1: Frozen base model
PHASE1_EPOCHS = 10
PHASE1_LEARNING_RATE = 0.001

# Phase 2: Fine-tuning
PHASE2_EPOCHS = 5
PHASE2_LEARNING_RATE = 0.0001
UNFREEZE_LAYERS = 20

# Data Augmentation
AUGMENTATION = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.15
}
```

### Model Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | 95.3% |
| Top-3 Accuracy | 98.7% |
| F1 Score | 0.94 |
| Precision | 0.96 |
| Recall | 0.93 |

### Training the Model (Optional)

If you want to retrain the model:
```bash
# 1. Download PlantVillage dataset
# Place in data/raw/plantvillage/color/

# 2. Prepare data splits
python scripts/setup_data.py

# 3. Train model
python scripts/run_training.py

# 4. Evaluate model
python scripts/run_evaluation.py
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

### 1. Fork the Repository
```bash
# Click "Fork" button on GitHub
```

### 2. Create Feature Branch
```bash
git checkout -b feature/amazing-feature
```

### 3. Commit Changes
```bash
git commit -m "feat: Add amazing feature"
```

**Commit Message Format:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

### 4. Push to Branch
```bash
git push origin feature/amazing-feature
```

### 5. Open Pull Request
- Go to GitHub and click "New Pull Request"
- Provide clear description of changes
- Reference any related issues

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint rules for JavaScript/React
- Write unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
```
MIT License

Copyright (c) 2025 Ashish Rathod

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Contact

### Project Maintainer
**Ashish Rathod**
- 🐙 GitHub: [@AshishRathodDev](https://github.com/AshishRathodDev)
- 📧 Email: ashish3110rathod@gmail.com
- 💼 LinkedIn: [Ashish Rathod](https://www.linkedin.com/in/ashishrathod-it/)
- 🌐 Portfolio: [ashishrathod.dev](https://ashishrathod.dev)

### Project Links
- 🔗 Repository: [https://github.com/AshishRathodDev/FarmAI-Analytics](https://github.com/AshishRathodDev/FarmAI-Analytics)
- 🐛 Issue Tracker: [GitHub Issues](https://github.com/AshishRathodDev/FarmAI-Analytics/issues)
- 📖 Documentation: [Wiki](https://github.com/AshishRathodDev/FarmAI-Analytics/wiki)

---

## 🙏 Acknowledgments

### Special Thanks To
- **PlantVillage Dataset** - For providing the comprehensive disease image dataset
- **Google Gemini AI** - For powering the intelligent chatbot assistant
- **TensorFlow Team** - For the robust deep learning framework
- **EfficientNet Authors** - For the efficient CNN architecture
- **React & Vite Teams** - For the excellent frontend tooling
- **Tailwind CSS** - For the utility-first CSS framework
- **Open Source Community** - For inspiration and support

### Research Papers
1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*
2. Hughes, D. P., & Salathe, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint*

### Technologies Used
- [TensorFlow](https://www.tensorflow.org/)
- [Flask](https://flask.palletsprojects.com/)
- [React](https://react.dev/)
- [Vite](https://vitejs.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Google Gemini AI](https://ai.google.dev/)
- [Lucide Icons](https://lucide.dev/)

---

## 📊 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/AshishRathodDev/FarmAI-Analytics?style=social)
![GitHub forks](https://img.shields.io/github/forks/AshishRathodDev/FarmAI-Analytics?style=social)
![GitHub issues](https://img.shields.io/github/issues/AshishRathodDev/FarmAI-Analytics)
![GitHub pull requests](https://img.shields.io/github/issues-pr/AshishRathodDev/FarmAI-Analytics)


---

## 📸 Screenshots

### Dashboard
![Dashboard](https://via.placeholder.com/800x500/10b981/ffffff?text=Dashboard+Screenshot)

### Disease Scanner
![Scanner](https://via.placeholder.com/800x500/10b981/ffffff?text=Scanner+Screenshot)

### AI Chatbot
![Chatbot](https://via.placeholder.com/800x500/10b981/ffffff?text=Chatbot+Screenshot)

### Analytics
![Analytics](https://via.placeholder.com/800x500/10b981/ffffff?text=Analytics+Screenshot)

---

## ⚠️ Disclaimer

This software is provided for educational and research purposes. While we strive for accuracy, FarmAI should be used as a supplementary tool alongside professional agricultural advice. Always consult with local agricultural experts before implementing treatment recommendations.

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AshishRathodDev/FarmAI-Analytics&type=Date)](https://star-history.com/#AshishRathodDev/FarmAI-Analytics&Date)

---

<div align="center">

**Made with ❤️ by [Ashish Rathod](https://github.com/AshishRathodDev)**

**If you find this project helpful, please consider giving it a ⭐!**

[⬆ Back to Top](#-farmai-analytics---ai-powered-crop-disease-detection-platform)

</div>
