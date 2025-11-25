"""
FarmAI Flask API - Render+Vercel Production Ready (Full CORS & Memory Optimized)
"""
import os
import sys
import json
import logging
import gc
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import keras
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CORS config to allow requests from Vercel prod site
CORS(app,
     resources={r"/*": {"origins": [
         "https://farm-ai-ten.vercel.app",
         "https://vercel.com",
         "*",  # Debug mode, remove in production for extra safety
     ]}}
)

@app.after_request
def after_request(response):
    # Always add CORS headers to response
    allowed = [
        "https://farm-ai-ten.vercel.app",
        "https://vercel.com"
    ]
    origin = request.headers.get('Origin')
    if origin in allowed:
        response.headers.add('Access-Control-Allow-Origin', origin)
    else:
        response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        response.headers.add('Access-Control-Max-Age', '3600')
        return response, 200

UPLOAD_FOLDER = 'uploads'
MODEL_FILENAME = 'crop_disease_classifier_final.keras'
CLASS_INDICES_FILENAME = 'class_indices.json'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, CLASS_INDICES_FILENAME)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
HUGGINGFACE_REPO = "rathodashish10/farmai-models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

model = None
class_names = []
model_loaded = False

def download_from_huggingface():
    try:
        logger.info("="*60)
        logger.info("🤗 Checking Hugging Face Models...")
        logger.info(f"Repository: {HUGGINGFACE_REPO}")
        logger.info("="*60)
        if not os.path.exists(MODEL_PATH):
            logger.info(f"📥 Downloading {MODEL_FILENAME}...")
            hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename=MODEL_FILENAME,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            logger.info(f"✅ Model downloaded successfully!")
        else:
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            logger.info(f"✅ Model already exists ({size_mb:.1f} MB)")
        if not os.path.exists(CLASS_INDICES_PATH):
            logger.info(f"📥 Downloading {CLASS_INDICES_FILENAME}...")
            hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename=CLASS_INDICES_FILENAME,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            logger.info(f"✅ Class indices downloaded successfully!")
        else:
            logger.info(f"✅ Class indices already exist")
        return True
    except Exception as e:
        logger.error(f"❌ Error downloading from Hugging Face: {e}")
        logger.error("Check HF repo, filenames, or token!")
        return False

def load_model_and_classes():
    global model, class_names, model_loaded
    try:
        logger.info("="*60)
        logger.info("🚀 Initializing FarmAI Model...")
        logger.info("="*60)

        if not download_from_huggingface():
            logger.error("❌ Failed to download models from Hugging Face")
            return False
        
        logger.info(f"📖 Loading class indices from {CLASS_INDICES_PATH}...")
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        class_names = list(class_indices.keys())
        logger.info(f"✅ Loaded {len(class_names)} disease classes")

        logger.info(f"🤖 Loading Keras model from {MODEL_PATH}...")
        logger.info(f"Using Keras version: {keras.__version__}")
        gc.collect()  # Free up memory
        keras.backend.clear_session()
        model = keras.models.load_model(MODEL_PATH)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        _ = model.predict(np.zeros((1, 160, 160, 3)), verbose=0)  # Warm up
        model_loaded = True
        logger.info("✅ Model loaded successfully!")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        logger.info("="*60)
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        logger.exception("Full traceback:")
        model_loaded = False
        return False

def allowed_file(filename):
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((160, 160))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {e}")
        raise

def format_disease_name(disease_name):
    formatted = disease_name.replace('___', ': ')
    formatted = formatted.replace('_', ' ')
    return formatted.title()

@app.route('/', methods=['GET', 'OPTIONS'])
@app.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
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
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return response, 200
    try:
        logger.info("="*60)
        logger.info("PREDICT ENDPOINT CALLED")
        logger.info("="*60)
        if not model_loaded or model is None:
            logger.error("MODEL NOT LOADED!")
            return jsonify({'status': 'error', 'message': 'Model not loaded.'}), 503
        if 'file' not in request.files:
            logger.error("NO FILE IN REQUEST")
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        file = request.files['file']
        logger.info(f"File received: {file.filename}")
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Processing image: {filename}")
        img_array = preprocess_image(filepath)
        logger.info("Running prediction...")
        predictions = model.predict(img_array, verbose=0)[0]
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
        try:
            os.remove(filepath)
        except:
            pass
        logger.info(f"✅ Prediction: {top_3_predictions[0]['disease']} ({top_3_predictions[0]['confidence_percent']})")
        del img_array, file, predictions  # free RAM
        gc.collect()
        return jsonify({
            'status': 'success',
            'prediction': top_3_predictions[0]['disease'],
            'confidence': top_3_predictions[0]['confidence'],
            'confidence_percent': top_3_predictions[0]['confidence_percent'],
            'top_3': top_3_predictions
        }), 200
    except Exception as e:
        logger.error("="*60)
        logger.error("PREDICTION FAILED")
        logger.error(f"Error: {str(e)}")
        logger.exception("Full traceback:")
        logger.error("="*60)
        return jsonify({'status': 'error', 'message': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/classes', methods=['GET', 'OPTIONS'])
def get_classes():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return response, 200
    try:
        if not class_names:
            return jsonify({'status': 'error', 'message': 'Classes not loaded'}), 503
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
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(413)
def too_large(error):
    return jsonify({'status': 'error', 'message': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.exception("Internal server error")
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

with app.app_context():
    load_model_and_classes()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    logger.info("="*60)
    logger.info(f"Starting FarmAI Backend on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"CORS enabled for all origins")
    logger.info("="*60)
    app.run(host='0.0.0.0', port=port, debug=debug)

