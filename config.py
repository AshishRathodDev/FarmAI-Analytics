"""
FarmAI Analytics Platform - Configuration Module
Complete configuration management for production deployment
"""

# config.py (replace top section)
from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Ensure important directories exist before logging or other modules run
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

for p in (MODELS_DIR, LOGS_DIR, DATA_DIR, DATA_DIR / "raw", DATA_DIR / "processed"):
    p.mkdir(parents=True, exist_ok=True)

# Env & keys
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Paths
DATABASE_PATH = BASE_DIR / 'farmer_analytics.db'
DATABASE_BACKUP_PATH = BASE_DIR / 'farmer_analytics_backup.db'
MODEL_PATH = MODELS_DIR / 'crop_disease_classifier_final.h5'
MODEL_METADATA_PATH = MODELS_DIR / 'model_metadata.json'
CLASS_INDICES_PATH = MODELS_DIR / 'class_indices.json'

# Model params (keep your existing constants after this)
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.65
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001


# ==================== DATA CONFIGURATION ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
DATA_RAW_PATH = DATA_DIR / 'raw'
DATA_PROCESSED_PATH = DATA_DIR / 'processed'
DATASET_NAME = 'plantvillage'

# Create directories if they don't exist
for dir_path in [DATA_DIR, DATA_RAW_PATH, DATA_PROCESSED_PATH, Path('models'), Path('logs')]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== CHATBOT CONFIGURATION ====================
CHATBOT_TEMPERATURE = 0.7
CHATBOT_MAX_TOKENS = 800
CHATBOT_MODEL = 'gemini-pro'

# System prompt template
CHATBOT_SYSTEM_PROMPT = """You are an expert agricultural advisor helping Indian farmers.
You provide practical, actionable advice that is:
1. Easy to understand (simple language)
2. Based on Indian farming practices
3. Cost-effective solutions
4. Includes specific step-by-step guidance
5. Uses common/local pesticide brands when relevant

Always be empathetic and encouraging. Farmers depend on your advice for their livelihood."""

# ==================== ANALYTICS CONFIGURATION ====================
ANALYTICS_UPDATE_INTERVAL = 3600  # seconds
METRICS_CACHE_DURATION = 300  # 5 minutes

# ==================== SUPPORTED CROPS & DISEASES ====================
SUPPORTED_CROPS = [
    'Wheat', 'Rice', 'Cotton', 'Corn', 'Sugarcane',
    'Tomato', 'Potato', 'Pepper', 'Grape', 'Apple',
    'Soybean', 'Strawberry', 'Peach', 'Orange'
]

# Disease database (expandable)
DISEASE_CLASSES = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Grape___Black_rot',
    5: 'Grape___Esca_(Black_Measles)',
    6: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    7: 'Grape___healthy',
    8: 'Orange___Haunglongbing_(Citrus_greening)',
    9: 'Peach___Bacterial_spot',
    10: 'Peach___healthy',
    11: 'Pepper,_bell___Bacterial_spot',
    12: 'Pepper,_bell___healthy',
    13: 'Potato___Early_blight',
    14: 'Potato___Late_blight',
    15: 'Potato___healthy',
    16: 'Tomato___Bacterial_spot',
    17: 'Tomato___Early_blight',
    18: 'Tomato___Late_blight',
    19: 'Tomato___Leaf_Mold',
    20: 'Tomato___Septoria_leaf_spot',
    21: 'Tomato___Spider_mites_Two-spotted_spider_mite',
    22: 'Tomato___Target_Spot',
    23: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    24: 'Tomato___Tomato_mosaic_virus',
    25: 'Tomato___healthy'
}

# Crop-specific common diseases
CROP_DISEASES = {
    'wheat': ['Rust', 'Powdery Mildew', 'Septoria', 'Smut'],
    'rice': ['Blast', 'Leaf Scald', 'Sheath Blight', 'Brown Spot'],
    'cotton': ['Leaf Spot', 'Wilt', 'Bollworm', 'Root Rot'],
    'corn': ['Smut', 'Rust', 'Gray Leaf Spot', 'Northern Blight'],
    'tomato': ['Early Blight', 'Late Blight', 'Leaf Curl', 'Bacterial Spot'],
    'potato': ['Early Blight', 'Late Blight', 'Scab', 'Black Leg'],
}

# ==================== STREAMLIT CONFIGURATION ====================
# NOTE: These are Streamlit specific. As you are using React,
# these might not be directly used by your Flask backend,
# but can remain for project context if you desire.
PAGE_ICON = "🌾"
PAGE_TITLE = "FarmAI Analytics Platform"
PAGE_LAYOUT = "wide"
THEME = "light"

# ==================== LOGGING CONFIGURATION ====================
LOG_FILE = 'logs/app.log'
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ==================== CLOUD CONFIGURATION ====================
# AWS S3 (for production file storage)
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET', 'farmai-storage')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Firebase (optional - for mobile app integration)
FIREBASE_CONFIG = {
    'apiKey': os.getenv('FIREBASE_API_KEY'),
    'projectId': os.getenv('FIREBASE_PROJECT_ID'),
}

# ==================== FEATURE FLAGS ====================
ENABLE_VOICE_INPUT = False  # Voice query support (future)
ENABLE_MULTI_LANGUAGE = True  # Multi-language support
ENABLE_OFFLINE_MODE = False  # Offline model inference (future)
ENABLE_ANALYTICS_EXPORT = True  # Power BI/Tableau export

# ==================== LANGUAGE SUPPORT ====================
SUPPORTED_LANGUAGES = ['English', 'Hindi', 'Marathi', 'Tamil', 'Telugu']

LANGUAGE_CODES = {
    'English': 'en',
    'Hindi': 'hi',
    'Marathi': 'mr',
    'Tamil': 'ta',
    'Telugu': 'te'
}

# ==================== TREATMENT TEMPLATES ====================
TREATMENT_TEMPLATES = {
    'fungal': """
    Fungal Disease Treatment Plan:
    
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
    
    'bacterial': """
     Bacterial Disease Treatment Plan:
    
    Immediate Actions (Day 1-3):
    - Remove infected plants completely
    - Disinfect tools with 70% alcohol
    - Avoid working with wet plants
    
    Treatment (Week 1-3):
    - Apply copper-based bactericide
    - Spray every 5-7 days
    - Use antibiotic if severe
    
    Prevention:
    - Use certified disease-free seeds
    - Practice strict sanitation
    - Avoid overhead irrigation
    
    Cost: {cost} per acre
    Recovery Time: {recovery_days} days
    """,
    
    'viral': """
     Viral Disease Management Plan:
    
    Immediate Actions (Day 1):
    - Remove infected plants immediately
    - Control insect vectors (aphids, whiteflies)
    - Isolate healthy plants
    
    Management (Ongoing):
    - No cure available - focus on prevention
    - Apply insecticides to control vectors
    - Use reflective mulches
    
    Prevention:
    - Plant resistant varieties
    - Use virus-free seeds
    - Control weed hosts
    
    Cost: {cost} per acre
    Note: Prevention is key - no cure exists
    """
}

# ==================== VALIDATION ====================
def validate_config():
    """Validate critical configuration settings"""
    errors = []
    warnings = []
    
    # Critical checks
    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is missing")
    
    if not Path(MODEL_PATH).parent.exists():
        warnings.append(f"Model directory doesn't exist: {Path(MODEL_PATH).parent}")
    
    if not DATABASE_PATH:
        errors.append("DATABASE_PATH not configured")
    
    # Return status
    if errors:
        return False, errors, warnings
    return True, [], warnings

# Run validation
is_valid, errors, warnings = validate_config()

if not is_valid:
    print("\n  CONFIGURATION ERRORS:")
    for error in errors:
        print(f"    {error}")

if warnings:
    print("\nWARNINGS:")
    for warning in warnings:
        print(f"     {warning}")

# ==================== EXPORT ALL ====================
__all__ = [
    'GOOGLE_API_KEY',
    'DATABASE_PATH',
    'MODEL_PATH',
    'CLASS_INDICES_PATH', # <-- ADDED THIS LINE
    'IMAGE_SIZE',
    'SUPPORTED_CROPS',
    'DISEASE_CLASSES',
    'PAGE_ICON', # Streamlit specific, but can remain if you want
    'PAGE_TITLE', # Streamlit specific
    'LOG_FILE',
    'CONFIDENCE_THRESHOLD',
    'CHATBOT_MODEL',
    'CHATBOT_SYSTEM_PROMPT',
    'SUPPORTED_LANGUAGES',
    'TREATMENT_TEMPLATES'
]