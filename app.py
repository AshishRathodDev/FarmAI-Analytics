"""
FarmAI Flask API - Production Ready with CORS FIX
Auto-downloads model from Hugging Face on startup
"""

import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ============================================================================
# CRITICAL: Import Keras before TensorFlow to use Keras 3
# ============================================================================
import keras
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Flask App Configuration
# ============================================================================
app = Flask(__name__)

# ============================================================================
# FIXED: Complete CORS Configuration - Production Ready
# ============================================================================
CORS(app, 
     resources={r"/*": {
         "origins": ["*"],  # Allow all origins for now
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Accept", "Authorization"],
         "expose_headers": ["Content-Type"],
         "supports_credentials": False,
         "max_age": 3600
     }},
     send_wildcard=True,
     allow_headers=["Content-Type", "Accept", "Authorization"],
     methods=["GET", "POST", "OPTIONS"]
)

# ============================================================================
# ADDED: Global CORS headers on all responses
# ============================================================================
@app.after_request
def after_request(response):
    """Add CORS headers to every response"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

# ============================================================================
# FIXED: Handle preflight OPTIONS requests
# ============================================================================
@app.before_request
def handle_preflight():
    """Handle OPTIONS preflight requests"""
    if request.method == "OPTIONS":
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        response.headers.add('Access-Control-Max-Age', '3600')
        return response, 200

# ============================================================================
# Configuration Constants
# ============================================================================
UPLOAD_FOLDER = 'uploads'
MODEL_FILENAME = 'crop_disease_classifier_final.keras'
CLASS_INDICES_FILENAME = 'class_indices.json'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, CLASS_INDICES_FILENAME)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
HUGGINGFACE_REPO = "rathodashish10/farmai-models"

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Flask config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# ============================================================================
# Global Variables
# ============================================================================
model = None
class_names = []
model_loaded = False

# ============================================================================
# Helper Functions
# ============================================================================

def download_from_huggingface():
    """Download model and class indices from Hugging Face"""
    try:
        logger.info("="*60)
        logger.info("Checking Hugging Face Models...")
        logger.info(f"Repository: {HUGGINGFACE_REPO}")
        logger.info("="*60)
        
        # Download model if not exists
        if not os.path.exists(MODEL_PATH):
            logger.info(f"Downloading {MODEL_FILENAME}...")
            hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename=MODEL_FILENAME,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            logger.info(f"Model downloaded successfully!")
        else:
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            logger.info(f"Model already exists ({size_mb:.1f} MB)")
        
        # Download class indices if not exists
        if not os.path.exists(CLASS_INDICES_PATH):
            logger.info(f"Downloading {CLASS_INDICES_FILENAME}...")
            hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename=CLASS_INDICES_FILENAME,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            logger.info(f"Class indices downloaded successfully!")
        else:
            logger.info(f"Class indices already exist")
        
        return True
    
    except Exception as e:
        logger.error(f"Error downloading from Hugging Face: {e}")
        logger.error(f"Please check:")
        logger.error(f"  1. Repository exists: {HUGGINGFACE_REPO}")
        logger.error(f"  2. Files exist in repo: {MODEL_FILENAME}, {CLASS_INDICES_FILENAME}")
        logger.error(f"  3. Repository is public or HF token is set")
        return False

def load_model_and_classes():
    """Load ML model and class names"""
    global model, class_names, model_loaded
    
    try:
        logger.info("="*60)
        logger.info("Initializing FarmAI Model...")
        logger.info("="*60)
        
        # Download from Hugging Face if needed
        if not download_from_huggingface():
            logger.error("Failed to download models from Hugging Face")
            return False
        
        # Load class names first
        logger.info(f"Loading class indices from {CLASS_INDICES_PATH}...")
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        class_names = list(class_indices.keys())
        logger.info(f"Loaded {len(class_names)} disease classes")
        
        # Load model
        logger.info(f"Loading Keras model from {MODEL_PATH}...")
        logger.info(f"Using Keras version: {keras.__version__}")
        
        model = keras.models.load_model(MODEL_PATH)
        model_loaded = True
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        logger.info("="*60)
        
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.exception("Full traceback:")
        model_loaded = False
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """
    Preprocess image for model prediction
    Resize to 160x160 and normalize to [0,1]
    """
    try:
        # Open and convert to RGB
        img = Image.open(img_path).convert('RGB')
        
        # Resize to model's expected input size
        img = img.resize((160, 160))
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def format_disease_name(disease_name):
    """Format disease name for better readability"""
    # Replace underscores and format
    formatted = disease_name.replace('___', ': ')
    formatted = formatted.replace('_', ' ')
    return formatted.title()

# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/', methods=['GET', 'OPTIONS'])
@app.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return response, 200
        
    response = jsonify({
        'status': 'healthy' if model_loaded else 'starting',
        'message': 'FarmAI API is running!',
        'model_loaded': model_loaded,
        'classes_count': len(class_names),
        'keras_version': keras.__version__,
        'huggingface_repo': HUGGINGFACE_REPO,
        'cors_enabled': True,
        'endpoints': {
            'health': '/',
            'predict': '/api/predict',
            'classes': '/api/classes'
        }
    })
    return response, 200

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    Disease prediction endpoint
    Accepts: multipart/form-data with 'file' field
    Returns: JSON with top 3 predictions
    """
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return response, 200
    
    try:
        # Check if model is loaded
        if not model_loaded or model is None:
            logger.warning("Prediction request but model not loaded")
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded. Please wait for initialization.'
            }), 503
        
        # Validate file in request
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        # Check if file selected
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing image: {filename}")
        
        # Preprocess and predict
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_predictions = [
            {
                'disease': format_disease_name(class_names[idx]),
                'confidence': float(predictions[idx]),
                'confidence_percent': f"{float(predictions[idx]) * 100:.2f}%",
                'raw_name': class_names[idx]
            }
            for idx in top_3_indices
        ]
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        logger.info(f"Prediction: {top_3_predictions[0]['disease']} ({top_3_predictions[0]['confidence_percent']})")
        
        return jsonify({
            'status': 'success',
            'prediction': top_3_predictions[0]['disease'],
            'confidence': top_3_predictions[0]['confidence'],
            'confidence_percent': top_3_predictions[0]['confidence_percent'],
            'top_3': top_3_predictions
        }), 200
    
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/classes', methods=['GET', 'OPTIONS'])
def get_classes():
    """Get all available disease classes"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return response, 200
        
    try:
        if not class_names:
            return jsonify({
                'status': 'error',
                'message': 'Classes not loaded'
            }), 503
        
        formatted_classes = [
            {
                'raw_name': name,
                'display_name': format_disease_name(name)
            }
            for name in class_names
        ]
        
        return jsonify({
            'status': 'success',
            'classes': formatted_classes,
            'count': len(formatted_classes)
        }), 200
    
    except Exception as e:
        logger.exception("Error getting classes")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(413)
def too_large(error):
    return jsonify({
        'status': 'error',
        'message': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    logger.exception("Internal server error")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

# ============================================================================
# Startup
# ============================================================================

# Load model immediately on import
with app.app_context():
    load_model_and_classes()

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info("="*60)
    logger.info(f"Starting FarmAI Backend on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"CORS enabled for all origins")
    logger.info("="*60)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
