"""
FarmAI - OPTIMIZED Configuration for 8GB RAM / M1 Mac / CPU Training
This config is designed to prevent kernel crashes and memory issues
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "plantvillage" / "color"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, 
                  FIGURES_DIR, METRICS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION - OPTIMIZED FOR LOW MEMORY
# ============================================================================
MODEL_ARCHITECTURE = "EfficientNetB0"
USE_PRETRAINED_WEIGHTS = True
FREEZE_BASE_MODEL = True

# Image configuration - REDUCED for memory
IMG_SIZE = (160, 160)  #  REDUCED from 224x224 to save 50% memory
IMG_CHANNELS = 3
INPUT_SHAPE = IMG_SIZE + (IMG_CHANNELS,)

# Training hyperparameters - OPTIMIZED
BATCH_SIZE = 4  #  REDUCED from 8 to prevent memory overflow
RANDOM_SEED = 42

# Phase 1: Training with frozen base - MINIMAL EPOCHS
PHASE1_EPOCHS = 2 #  Keep minimal for testing
PHASE1_LEARNING_RATE = 0.001

# Phase 2: Fine-tuning - DISABLED for initial training
PHASE2_EPOCHS = 2  #  SET TO 0 to skip fine-tuning and save memory
PHASE2_LEARNING_RATE = 0.0001
UNFREEZE_LAYERS = 20  # Reduced from 30

# Model architecture details - REDUCED
DENSE_UNITS = 128  #  REDUCED from 256
DROPOUT_RATE = 0.3
ACTIVATION = 'relu'
FINAL_ACTIVATION = 'softmax'

# Compilation settings
LOSS_FUNCTION = 'categorical_crossentropy'
METRICS = ['accuracy']

# ============================================================================
# DATA AUGMENTATION - SIMPLIFIED
# ============================================================================
AUGMENTATION_CONFIG = {
    'rotation_range': 15,  # Reduced
    'width_shift_range': 0.1,  # Reduced
    'height_shift_range': 0.1,  # Reduced
    'horizontal_flip': True,
    'vertical_flip': False,
    'zoom_range': 0.1,  # Reduced
    'shear_range': 0.05,  # Reduced
    'fill_mode': 'nearest'
}

# ============================================================================
# DATASET SPLIT RATIOS
# ============================================================================
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

#  CRITICAL: Use only 3 classes for minimal memory footprint
NUM_CLASSES_TO_USE = 5  

# ============================================================================
# CALLBACKS CONFIGURATION
# ============================================================================
MONITOR_METRIC = 'val_loss'
EARLY_STOPPING_PATIENCE = 3  # Reduced
EARLY_STOPPING_RESTORE_BEST = True

REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 2  # Reduced
REDUCE_LR_MIN_LR = 1e-7

CHECKPOINT_MODE = 'min'
SAVE_BEST_ONLY = True

# ============================================================================
# FILE PATHS
# ============================================================================
FINAL_MODEL_PATH = MODELS_DIR / "crop_disease_classifier_final.keras"
CHECKPOINT_FILEPATH = MODELS_DIR / "best_model_checkpoint.h5"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"

TRAINING_HISTORY_PATH = METRICS_DIR / "training_history.json"
TRAINING_CURVES_FIGURE = FIGURES_DIR / "training_curves.png"

CLASSIFICATION_REPORT_PATH = METRICS_DIR / "classification_report.csv"
EVALUATION_METRICS_PATH = METRICS_DIR / "evaluation_metrics.txt"
CONFUSION_MATRIX_FIGURE = FIGURES_DIR / "confusion_matrix.png"
PREDICTIONS_FIGURE = FIGURES_DIR / "sample_predictions.png"
GRADCAM_FIGURE = FIGURES_DIR / "gradcam_visualizations.png"

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
CM_FIGSIZE = (8, 6)  # Reduced
CM_CMAP = 'Blues'
CM_ANNOT_FORMAT = '.1f'
GRADCAM_NUM_SAMPLES = 2  # Reduced

# ============================================================================
# VERBOSITY
# ============================================================================
VERBOSE = 1  # Keep progress bar

# ============================================================================
# API CONFIGURATION
# ============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CONFIDENCE_THRESHOLD = 0.65

# ============================================================================
# MEMORY OPTIMIZATION FLAGS
# ============================================================================
#  NEW: Memory optimization settings
USE_MIXED_PRECISION = False  # Disable for CPU
CLEAR_SESSION_BETWEEN_PHASES = True  # Clear Keras session to free memory
USE_GENERATOR_WORKERS = 1  # Single worker to save memory
USE_GENERATOR_MULTIPROCESSING = False  # Disable multiprocessing

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================
def validate_config():
    """Validate configuration and check for common issues"""
    errors = []
    warnings = []
    
    if not RAW_DATA_DIR.exists():
        errors.append(f"Raw data directory not found: {RAW_DATA_DIR}")
    
    if abs((TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT) - 1.0) > 0.01:
        errors.append(f"Split ratios must sum to 1.0")
    
    if PHASE1_EPOCHS < 1:
        warnings.append("PHASE1_EPOCHS is less than 1")
    
    # Memory warnings
    if BATCH_SIZE > 4:
        warnings.append("  BATCH_SIZE > 4 may cause memory issues on 8GB RAM")
    
    if NUM_CLASSES_TO_USE > 5:
        warnings.append("  NUM_CLASSES_TO_USE > 5 may cause memory issues")
    
    return len(errors) == 0, errors, warnings

def get_config_summary():
    """Print configuration summary"""
    print("\n" + "="*70)
    print("⚡ OPTIMIZED CONFIG FOR LOW MEMORY (8GB RAM / CPU)")
    print("="*70)
    print(f"\n Project Directories:")
    print(f"  Base: {BASE_DIR}")
    print(f"  Raw Data: {RAW_DATA_DIR}")
    print(f"  Processed Data: {PROCESSED_DATA_DIR}")
    print(f"  Models: {MODELS_DIR}")
    
    print(f"\n Model Configuration (Memory Optimized):")
    print(f"  Architecture: {MODEL_ARCHITECTURE}")
    print(f"  Image Size: {IMG_SIZE} (reduced for memory)")
    print(f"  Batch Size: {BATCH_SIZE} (minimal for 8GB RAM)")
    print(f"  Classes: {NUM_CLASSES_TO_USE} (limited for memory)")
    print(f"  Dense Units: {DENSE_UNITS} (reduced)")
    
    print(f"\n Training Configuration:")
    print(f"  Phase 1 Epochs: {PHASE1_EPOCHS}")
    print(f"  Phase 2 Epochs: {PHASE2_EPOCHS} (disabled to save memory)")
    print(f"  Phase 1 LR: {PHASE1_LEARNING_RATE}")
    
    print(f"\n Data Splits:")
    print(f"  Train: {TRAIN_SPLIT*100:.0f}%")
    print(f"  Val: {VAL_SPLIT*100:.0f}%")
    print(f"  Test: {TEST_SPLIT*100:.0f}%")
    
    print(f"\n Memory Optimizations:")
    print(f"  Clear session between phases: {CLEAR_SESSION_BETWEEN_PHASES}")
    print(f"  Generator workers: {USE_GENERATOR_WORKERS}")
    print(f"  Multiprocessing: {USE_GENERATOR_MULTIPROCESSING}")
    
    print("="*70 + "\n")

__all__ = [
    'BASE_DIR', 'DATA_DIR', 'RAW_DATA_DIR', 'PROCESSED_DATA_DIR',
    'MODELS_DIR', 'RESULTS_DIR', 'FIGURES_DIR', 'METRICS_DIR',
    'MODEL_ARCHITECTURE', 'IMG_SIZE', 'BATCH_SIZE', 'RANDOM_SEED',
    'PHASE1_EPOCHS', 'PHASE2_EPOCHS', 'FINAL_MODEL_PATH', 
    'validate_config', 'get_config_summary', 'NUM_CLASSES_TO_USE',
    'CLEAR_SESSION_BETWEEN_PHASES', 'USE_GENERATOR_WORKERS'
]
