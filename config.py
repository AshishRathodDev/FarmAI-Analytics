"""
FarmAI Analytics Platform - Configuration Module
Centralized configuration for local dev and production deployments.
"""

from pathlib import Path
import os
from dotenv import load_dotenv
from typing import Tuple, List

# --- Base directories and env loading ---------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    # still call load_dotenv() to pick up env from process if set
    load_dotenv()

# Ensure important directories exist early
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
DATA_RAW_PATH = DATA_DIR / "raw"
DATA_PROCESSED_PATH = DATA_DIR / "processed"

for d in (MODELS_DIR, LOGS_DIR, DATA_DIR, DATA_RAW_PATH, DATA_PROCESSED_PATH):
    d.mkdir(parents=True, exist_ok=True)

# --- Environment / keys ----------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # may be None in local dev

# --- Paths (use Path objects everywhere) -----------------------------------
DATABASE_PATH = BASE_DIR / "farmer_analytics.db"
DATABASE_BACKUP_PATH = BASE_DIR / "farmer_analytics_backup.db"
MODEL_PATH = MODELS_DIR / "crop_disease_classifier_final.h5"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"

# --- Model / training defaults ---------------------------------------------
IMAGE_SIZE: Tuple[int, int] = (224, 224)
CONFIDENCE_THRESHOLD: float = 0.65
BATCH_SIZE: int = 32
EPOCHS: int = 30
LEARNING_RATE: float = 0.001

# --- Chatbot defaults ------------------------------------------------------
CHATBOT_TEMPERATURE = 0.7
CHATBOT_MAX_TOKENS = 800
CHATBOT_MODEL = "gemini-pro"

CHATBOT_SYSTEM_PROMPT = """You are an expert agricultural advisor helping Indian farmers.
You provide practical, actionable advice that is:
1. Easy to understand (simple language)
2. Based on Indian farming practices
3. Cost-effective solutions
4. Includes specific step-by-step guidance
5. Uses common/local pesticide brands when relevant

Always be empathetic and encouraging. Farmers depend on your advice for their livelihood."""

# --- Analytics / timing ----------------------------------------------------
ANALYTICS_UPDATE_INTERVAL = 3600  # seconds
METRICS_CACHE_DURATION = 300  # seconds

# --- Supported crops / disease classes (expandable) -------------------------
SUPPORTED_CROPS = [
    "Wheat", "Rice", "Cotton", "Corn", "Sugarcane",
    "Tomato", "Potato", "Pepper", "Grape", "Apple",
    "Soybean", "Strawberry", "Peach", "Orange"
]

DISEASE_CLASSES = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy",
    4: "Grape___Black_rot",
    5: "Grape___Esca_(Black_Measles)",
    6: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    7: "Grape___healthy",
    8: "Orange___Haunglongbing_(Citrus_greening)",
    9: "Peach___Bacterial_spot",
    10: "Peach___healthy",
    11: "Pepper,_bell___Bacterial_spot",
    12: "Pepper,_bell___healthy",
    13: "Potato___Early_blight",
    14: "Potato___Late_blight",
    15: "Potato___healthy",
    16: "Tomato___Bacterial_spot",
    17: "Tomato___Early_blight",
    18: "Tomato___Late_blight",
    19: "Tomato___Leaf_Mold",
    20: "Tomato___Septoria_leaf_spot",
    21: "Tomato___Spider_mites_Two-spotted_spider_mite",
    22: "Tomato___Target_Spot",
    23: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    24: "Tomato___Tomato_mosaic_virus",
    25: "Tomato___healthy",
}

CROP_DISEASES = {
    "wheat": ["Rust", "Powdery Mildew", "Septoria", "Smut"],
    "rice": ["Blast", "Leaf Scald", "Sheath Blight", "Brown Spot"],
    "cotton": ["Leaf Spot", "Wilt", "Bollworm", "Root Rot"],
    "corn": ["Smut", "Rust", "Gray Leaf Spot", "Northern Blight"],
    "tomato": ["Early Blight", "Late Blight", "Leaf Curl", "Bacterial Spot"],
    "potato": ["Early Blight", "Late Blight", "Scab", "Black Leg"],
}

# --- UI/streamlit context (optional, kept for project documentation) -------
PAGE_ICON = "🌾"
PAGE_TITLE = "FarmAI Analytics Platform"
PAGE_LAYOUT = "wide"
THEME = "light"

# --- Logging ---------------------------------------------------------------
LOG_FILE = LOGS_DIR / "app.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# --- Cloud / storage -------------------------------------------------------
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "farmai-storage")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

FIREBASE_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
}

# --- Feature flags --------------------------------------------------------
ENABLE_VOICE_INPUT = False
ENABLE_MULTI_LANGUAGE = True
ENABLE_OFFLINE_MODE = False
ENABLE_ANALYTICS_EXPORT = True

# --- Language support -----------------------------------------------------
SUPPORTED_LANGUAGES = ["English", "Hindi", "Marathi", "Tamil", "Telugu"]
LANGUAGE_CODES = {"English": "en", "Hindi": "hi", "Marathi": "mr", "Tamil": "ta", "Telugu": "te"}

# --- Treatment templates --------------------------------------------------
TREATMENT_TEMPLATES = {
    "fungal": """Fungal Disease Treatment Plan:
Immediate Actions (Day 1-3):
- Remove and destroy infected plant parts
- Improve air circulation around plants
- Avoid overhead watering

Treatment (Week 1-2):
- Apply fungicide: {fungicide_name}
- Spray every 7-10 days
- Ensure complete leaf coverage

Prevention:
- Use disease-resistant varieties
- Practice crop rotation
- Maintain proper plant spacing

Cost: {cost} per acre
Recovery Time: {recovery_days} days
""",
    "bacterial": """Bacterial Disease Treatment Plan:
Immediate Actions (Day 1-3):
- Remove infected plants completely
- Disinfect tools with 70% alcohol
- Avoid working with wet plants
...
""",
    "viral": """Viral Disease Management Plan:
Immediate Actions (Day 1):
- Remove infected plants immediately
- Control insect vectors (aphids, whiteflies)
- Isolate healthy plants
...
""",
}

# --- Validation ------------------------------------------------------------
def validate_config() -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    # API key is recommended for production but not fatal in dev
    if not GOOGLE_API_KEY:
        warnings.append("GOOGLE_API_KEY is not set; chatbot will run in fallback mode.")

    # Model file / directory
    if not MODEL_PATH.parent.exists():
        warnings.append(f"Model directory does not exist: {MODEL_PATH.parent}")

    # DB parent should exist (we create it above), but sanity-check permission
    if not DATABASE_PATH.parent.exists():
        warnings.append(f"Database directory does not exist: {DATABASE_PATH.parent}")

    # Return status: errors are fatal (none currently), warnings are informational
    if errors:
        return False, errors, warnings
    return True, [], warnings


# Run validation on import (prints friendly messages)
_is_valid, _errors, _warnings = validate_config()
if _errors:
    print("\nCONFIG ERRORS:")
    for e in _errors:
        print(" -", e)
if _warnings:
    print("\nCONFIG WARNINGS:")
    for w in _warnings:
        print(" -", w)

# --- Exports ---------------------------------------------------------------
__all__ = [
    "GOOGLE_API_KEY",
    "DATABASE_PATH",
    "DATABASE_BACKUP_PATH",
    "MODEL_PATH",
    "MODEL_METADATA_PATH",
    "CLASS_INDICES_PATH",
    "IMAGE_SIZE",
    "CONFIDENCE_THRESHOLD",
    "BATCH_SIZE",
    "EPOCHS",
    "LEARNING_RATE",
    "CHATBOT_MODEL",
    "CHATBOT_SYSTEM_PROMPT",
    "SUPPORTED_CROPS",
    "DISEASE_CLASSES",
    "CROP_DISEASES",
    "LOG_FILE",
    "TREATMENT_TEMPLATES",
    "SUPPORTED_LANGUAGES",
]
