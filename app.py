"""
FarmAI Simple Flask API with Hugging Face Model Auto-Download
"""

import os
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FILENAME = 'crop_disease_classifier_final.h5'
CLASS_INDICES_FILENAME = 'class_indices.json'
MODEL_PATH = f'models/{MODEL_FILENAME}'
CLASS_INDICES_PATH = f'models/{CLASS_INDICES_FILENAME}'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global model variables
model = None
class_names = []

def download_from_huggingface():
    """Download model from Hugging Face if not exists"""
    try:
        repo_id = "rathodashish10/farmai-models"
        
        # Download model if not exists
        if not os.path.exists(MODEL_PATH):
            logger.info(f" Downloading {MODEL_FILENAME} from Hugging Face...")
            hf_hub_download(
                repo_id=repo_id,
                filename=MODEL_FILENAME,
                local_dir="models",
                local_dir_use_symlinks=False
            )
            logger.info(f" Model downloaded successfully!")
        else:
            logger.info(f" Model already exists locally")
        
        # Download class indices if not exists
        if not os.path.exists(CLASS_INDICES_PATH):
            logger.info(f" Downloading {CLASS_INDICES_FILENAME} from Hugging Face...")
            hf_hub_download(
                repo_id=repo_id,
                filename=CLASS_INDICES_FILENAME,
                local_dir="models",
                local_dir_use_symlinks=False
            )
            logger.info(f" Class indices downloaded successfully!")
        else:
            logger.info(f" Class indices already exist locally")
        
        return True
    
    except Exception as e:
        logger.error(f" Error downloading from Hugging Face: {e}")
        return False

def load_model():
    """Load ML model at startup"""
    global model, class_names
    
    try:
        # Download from Hugging Face if needed
        if not download_from_huggingface():
            logger.warning("Failed to download models from Hugging Face")
            return False
        
        # Load model
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(" Model loaded successfully!")
        
        # Load class names
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        class_names = list(class_indices.keys())
        logger.info(f" Loaded {len(class_names)} disease classes")
        
        return True
    
    except Exception as e:
        logger.error(f" Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess image for model prediction"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((160, 160))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def format_disease_name(disease_name):
    """Format disease name for display"""
    return disease_name.replace('___', ': ').replace('_', ' ').title()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'FarmAI API is running!',
        'model_loaded': model is not None,
        'classes_loaded': len(class_names),
        'huggingface_repo': 'rathodashish10/farmai-models'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Disease prediction endpoint"""
    try:
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 503
        
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type. Allowed: png, jpg, jpeg'
            }), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array, verbose=0)[0]
        
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_predictions = [
            {
                'disease': format_disease_name(class_names[idx]),
                'confidence': float(predictions[idx]),
                'raw_name': class_names[idx]
            }
            for idx in top_3_indices
        ]
        
        os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'prediction': top_3_predictions[0]['disease'],
            'confidence': top_3_predictions[0]['confidence'],
            'top_3': top_3_predictions
        })
    
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get all disease classes"""
    try:
        if not class_names:
            return jsonify({
                'status': 'error',
                'message': 'Classes not loaded'
            }), 503
        
        formatted_classes = [format_disease_name(name) for name in class_names]
        
        return jsonify({
            'status': 'success',
            'classes': formatted_classes,
            'count': len(formatted_classes)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Not found'}), 404

@app.errorhandler(413)
def too_large(error):
    return jsonify({'status': 'error', 'message': 'File too large (max 10MB)'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal error'}), 500

# Initialize on startup
with app.app_context():
    logger.info("="*60)
    logger.info("🌾 FarmAI Backend Starting...")
    logger.info("="*60)
    
    model_loaded = load_model()
    if not model_loaded:
        logger.warning("️  Model not loaded!")
    else:
        logger.info(" All resources initialized!")
        logger.info("="*60)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    logger.info(f" Starting server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
