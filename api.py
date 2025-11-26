
##api.py    

"""
FarmAI Simple Flask API
Works directly with your existing src/ modules
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import keras
import numpy as np
from PIL import Image

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'crop_disease_classifier_final.keras'
CLASS_INDICES_PATH = 'models/class_indices.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global model and class names
model = None
class_names = []

def load_model():
    """Load ML model at startup"""
    global model, class_names
    try:
        logger.info(f"Checking for model at {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model not found at {MODEL_PATH}")
            return False

        logger.info(f"Loading model from {MODEL_PATH}...")
        model = keras.models.load_model(MODEL_PATH)
        logger.info("✅ Model loaded successfully!")

        # Load class names
        if os.path.exists(CLASS_INDICES_PATH):
            with open(CLASS_INDICES_PATH, 'r') as f:
                class_indices = json.load(f)
            class_names = list(class_indices.keys())
            logger.info(f"Loaded {len(class_names)} disease classes")
        else:
            logger.warning(f"Class indices not found at {CLASS_INDICES_PATH}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess image for prediction"""
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
    return disease_name.replace('___', ': ').replace('_', ' ').title()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'FarmAI API is running!',
        'model_loaded': model is not None,
        'classes_loaded': len(class_names)
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
                'message': 'Invalid file type'
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
    return jsonify({'status': 'error', 'message': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal error'}), 500

# Model load on app startup
with app.app_context():
    logger.info("="*60)
    logger.info("FarmAI Backend Starting...")
    logger.info("="*60)
    model_loaded = load_model()
    if not model_loaded:
        logger.warning("Model not loaded!")
    else:
        logger.info("All resources initialized!")

if __name__ == '__main__':
    logger.info("Starting server on http://0.0.0.0:5050")
    app.run(host='0.0.0.0', port=5050, debug=False)


