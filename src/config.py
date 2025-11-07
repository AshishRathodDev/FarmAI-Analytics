"""
Configuration file for Syngenta Crop Disease Classification


"""

import os
from pathlib import Path


# PROJECT PATHS


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Adjust this path based on your actual dataset structure

RAW_DATA_DIR = DATA_DIR / "raw" / "plantvillage" / "color"



PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create directories if they don't exist    
for dir_path in [PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, 
                 FIGURES_DIR, METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)



# DATA CONFIGURATION

# Dataset splits
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Data loading
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
AUTOTUNE = -1  # tf.data.AUTOTUNE

# Class selection (use all or subset)
# Set to None to use all classes, or specify number like 12, 15
NUM_CLASSES_TO_USE = 12  # For faster training, use subset


# MODEL CONFIGURATION


# Model architecture
MODEL_ARCHITECTURE = "EfficientNetB0"  # Options: "EfficientNetB0", "ResNet50", "MobileNetV2"
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Transfer learning
USE_PRETRAINED_WEIGHTS = True
FREEZE_BASE_MODEL = True  # Initially freeze base model

# Model architecture details
DENSE_UNITS = 256
DROPOUT_RATE = 0.3
ACTIVATION = 'relu'
FINAL_ACTIVATION = 'softmax'


# TRAINING CONFIGURATION


# Phase 1: Train with frozen base
PHASE1_EPOCHS = 15
PHASE1_LEARNING_RATE = 0.001

# Phase 2: Fine-tuning with unfrozen layers
PHASE2_EPOCHS = 15
PHASE2_LEARNING_RATE = 1e-5
UNFREEZE_LAYERS = 30  # Number of layers to unfreeze from top

# Optimizer
OPTIMIZER = 'adam'

# Loss function
LOSS_FUNCTION = 'categorical_crossentropy'

# Metrics
METRICS = ['accuracy']


# CALLBACKS CONFIGURATION


# Model checkpoint
CHECKPOINT_FILEPATH = str(MODELS_DIR / "best_crop_disease_model.h5")
SAVE_BEST_ONLY = True
MONITOR_METRIC = 'val_accuracy'
CHECKPOINT_MODE = 'max'

# Early stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_RESTORE_BEST = True

# Learning rate reduction
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_MIN_LR = 1e-7


# AUGMENTATION CONFIGURATION


# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 30,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': True,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest'
}


# EVALUATION CONFIGURATION


# Confusion matrix
CM_FIGSIZE = (12, 10)
CM_CMAP = 'Blues'
CM_ANNOT_FORMAT = '.1f'

# Grad-CAM
GRADCAM_LAYER_NAME = None  # Will be auto-detected
GRADCAM_NUM_SAMPLES = 5


# REPRODUCIBILITY

# Random seeds for reproducibility
RANDOM_SEED = 42

# ============================================================================
# LOGGING
# ============================================================================

# Verbosity
VERBOSE = 1


# DEMO APP CONFIGURATION


# Gradio app settings
DEMO_PORT = 7860
DEMO_SHARE = False  # Set to True for public link
DEMO_EXAMPLES_DIR = DATA_DIR / "raw" / "plantvillage" / "color"


# OUTPUT FILES


# Model files
FINAL_MODEL_PATH = MODELS_DIR / "crop_disease_classifier_final.h5"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"

# Result files
TRAINING_HISTORY_PATH = METRICS_DIR / "training_history.json"
CLASSIFICATION_REPORT_PATH = METRICS_DIR / "classification_report.csv"
EVALUATION_METRICS_PATH = METRICS_DIR / "evaluation_metrics.txt"

# Figure files
CLASS_DIST_FIGURE = FIGURES_DIR / "class_distribution.png"
SAMPLE_IMAGES_FIGURE = FIGURES_DIR / "sample_images.png"
TRAINING_CURVES_FIGURE = FIGURES_DIR / "training_curves.png"
CONFUSION_MATRIX_FIGURE = FIGURES_DIR / "confusion_matrix.png"
GRADCAM_FIGURE = FIGURES_DIR / "gradcam_examples.png"
PREDICTIONS_FIGURE = FIGURES_DIR / "sample_predictions.png"


# VALIDATION


def validate_config():
    """Validate configuration settings"""
    
    # Check splits sum to 1.0
    total_split = TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT
    assert abs(total_split - 1.0) < 0.01, f"Splits must sum to 1.0, got {total_split}"
    
    # Check data directory exists
    assert RAW_DATA_DIR.exists(), f"Raw data directory not found: {RAW_DATA_DIR}"
    
    # Check architecture is valid
    valid_architectures = ["EfficientNetB0", "ResNet50", "MobileNetV2"]
    assert MODEL_ARCHITECTURE in valid_architectures, \
        f"Invalid architecture. Choose from: {valid_architectures}"
    
    print("✓ Configuration validated successfully")
    return True


# HELPER FUNCTIONS


def get_config_summary():
    """Print configuration summary"""
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Raw Data: {RAW_DATA_DIR}")
    print(f"\nModel Architecture: {MODEL_ARCHITECTURE}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"\nPhase 1: {PHASE1_EPOCHS} epochs, LR={PHASE1_LEARNING_RATE}")
    print(f"Phase 2: {PHASE2_EPOCHS} epochs, LR={PHASE2_LEARNING_RATE}")
    print(f"\nRandom Seed: {RANDOM_SEED}")
    print("="*70 + "\n")

if __name__ == "__main__":
    validate_config()
    get_config_summary()
    