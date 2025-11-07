# 🌾 Syngenta Crop Disease Classification

**AI-Powered Plant Disease Detection System**

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

1. Download PlantVillage dataset from Kaggle
2. Extract and place in `data/raw/plantvillage/color/`

### 3. Run Complete Pipeline

```bash
# Setup project structure
python scripts/setup_project.py

# Prepare data (splits into train/val/test)
python scripts/setup_data.py

# Run training
python scripts/run_training.py

# Run evaluation
python scripts/run_evaluation.py

# Launch demo app
python demo/app_gradio.py
```

---

## 📂 Project Structure

```
syngenta_crop_disease/
├── data/                    ← Datasets
├── src/                     ← Python modules
├── notebooks/               ← Jupyter notebooks
├── models/                  ← Trained models
├── results/                 ← Outputs
├── demo/                    ← Gradio app
├── deliverables/            ← Final submission
└── scripts/                 ← Automation scripts
```

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| Accuracy | 95%+ |
| Model Size | ~15 MB |
| Inference Time | <100ms |
| Classes | 12 diseases |

---

## 📊 Deliverables

- ✅ Trained Model (`best_crop_disease_model.h5`)
- ✅ Evaluation Metrics (CSV reports)
- ✅ Confusion Matrix (PNG)
- ✅ Grad-CAM Visualizations
- ✅ Gradio Demo App
- ✅ Manager Report
- ✅ Presentation Slides

---

## 🛠️ Development

### Run Notebooks

```bash
jupyter notebook notebooks/
```

### Test Individual Modules

```bash
# Test data pipeline
python src/data_utils.py

# Test model
python src/model.py
```

---

## 📝 Citation

Dataset: PlantVillage  
Model: EfficientNetB0 (Transfer Learning)  
Framework: TensorFlow/Keras

---

For detailed documentation, see `deliverables/README.md`a