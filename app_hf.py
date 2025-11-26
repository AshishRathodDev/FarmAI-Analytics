"""
FarmAI Flask API - FINAL WORKING VERSION
Optimized for Cloud Run (Signal Timeout Removed to fix 500 Error)
"""

import os
import sys
import json
import logging
import gc
# import signal  <-- REMOVED THIS
# from contextlib import contextmanager <-- REMOVED THIS
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Force CPU-only BEFORE any TF/Keras imports
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'

import keras
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CORS
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
    return response

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response('', 204)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
        return response

# Config
UPLOAD_FOLDER = 'uploads'
MODEL_FILENAME = 'crop_disease_classifier_final.keras'
CLASS_INDICES_FILENAME = 'class_indices.json'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, CLASS_INDICES_FILENAME)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
MAX_FILE_SIZE = 5 * 1024 * 1024
HUGGINGFACE_REPO = "rathodashish10/farmai-models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

model = None
class_names = []
model_loaded = False

def download_from_huggingface():
    """Download model files"""
    try:
        logger.info("="*60)
        logger.info("Downloading from Hugging Face...")
        
        if not os.path.exists(MODEL_PATH):
            logger.info(f"Downloading {MODEL_FILENAME}...")
            hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename=MODEL_FILENAME,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            logger.info("Model downloaded")
        else:
            logger.info("Model exists")
        
        if not os.path.exists(CLASS_INDICES_PATH):
            logger.info(f"Downloading {CLASS_INDICES_FILENAME}...")
            hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename=CLASS_INDICES_FILENAME,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            logger.info("Classes downloaded")
        else:
            logger.info("Classes exist")
        
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def load_model_and_classes():
    """Load model with aggressive optimization"""
    global model, class_names, model_loaded
    
    try:
        logger.info("="*60)
        logger.info("LOADING MODEL (CPU-optimized)")
        logger.info("="*60)

        if not download_from_huggingface():
            logger.error("Download failed")
            return False
        
        # Load classes
        logger.info("Loading classes...")
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        class_names = list(class_indices.keys())
        logger.info(f"Loaded {len(class_names)} classes")

        # Clear memory
        logger.info("Clearing memory...")
        gc.collect()
        keras.backend.clear_session()
        
        # Load model
        logger.info(f"Loading model from {MODEL_PATH}...")
        logger.info(f"Keras: {keras.__version__}")
        
        model = keras.models.load_model(MODEL_PATH, compile=False)
        
        # Optimize for CPU
        logger.info("Compiling for CPU...")
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            run_eagerly=False
        )
        
        # Warm up
        logger.info("Warming up model...")
        dummy = np.zeros((1, 160, 160, 3), dtype=np.float32)
        
        # REMOVED SIGNAL TIMEOUT HERE
        try:
            _ = model.predict(dummy, verbose=0)
            logger.info("Warm-up complete")
        except Exception as e:
            logger.warning(f"Warm-up failed but continuing: {e}")
        
        del dummy
        gc.collect()
        
        model_loaded = True
        logger.info("✓ MODEL READY")
        logger.info(f"  Input: {model.input_shape}")
        logger.info(f"  Output: {model.output_shape}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.exception("Traceback:")
        model_loaded = False
        return False

def allowed_file(filename):
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Fast preprocessing"""
    try:
        logger.info(f"Preprocessing: {os.path.basename(img_path)}")
        
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((160, 160), Image.Resampling.BILINEAR)
            img_array = np.array(img, dtype=np.float32)
        
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Shape: {img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

def format_disease_name(disease_name):
    formatted = disease_name.replace('___', ': ')
    formatted = formatted.replace('_', ' ')
    return formatted.title()

# API Endpoints

@app.route('/', methods=['GET', 'OPTIONS'])
@app.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        'status': 'healthy' if model_loaded else 'starting',
        'message': 'FarmAI API - Production Ready',
        'model_loaded': model_loaded,
        'classes_count': len(class_names),
        'keras_version': keras.__version__,
        'cpu_only': True,
        'optimized': True,
        'cors_enabled': True,
        'endpoints': {
            'health': '/',
            'predict': '/api/predict',
            'classes': '/api/classes'
        }
    }), 200

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    
    filepath = None
    start_time = None
    
    try:
        import time
        start_time = time.time()
        
        logger.info("="*60)
        logger.info("PREDICTION REQUEST")
        logger.info("="*60)
        
        # Check model
        if not model_loaded or model is None:
            logger.error("Model not loaded")
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 503
        
        # Validate request
        if 'file' not in request.files:
            logger.error("No file")
            return jsonify({
                'status': 'error',
                'message': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        logger.info(f"File: {file.filename}")
        
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved: {filename}")
        
        # Preprocess
        logger.info("Preprocessing...")
        preprocess_start = time.time()
        img_array = preprocess_image(filepath)
        preprocess_time = time.time() - preprocess_start
        logger.info(f"Preprocess time: {preprocess_time:.2f}s")
        
        # Predict (WITHOUT SIGNAL TIMEOUT)
        logger.info("Running prediction...")
        predict_start = time.time()
        
        # Direct prediction call without timeout wrapper
        predictions = model.predict(img_array, verbose=0)[0]
        
        predict_time = time.time() - predict_start
        logger.info(f"Prediction time: {predict_time:.2f}s")
        
        # Process results
        logger.info("Processing results...")
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
        
        total_time = time.time() - start_time
        logger.info(f"✓ Result: {top_3_predictions[0]['disease']} ({top_3_predictions[0]['confidence_percent']})")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info("="*60)
        
        # Cleanup
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        
        del img_array, predictions
        gc.collect()
        
        return jsonify({
            'status': 'success',
            'prediction': top_3_predictions[0]['disease'],
            'confidence': top_3_predictions[0]['confidence'],
            'confidence_percent': top_3_predictions[0]['confidence_percent'],
            'top_3': top_3_predictions,
            'inference_time': f"{predict_time:.2f}s",
            'total_time': f"{total_time:.2f}s"
        }), 200
        
    except Exception as e:
        logger.error("="*60)
        logger.error(f"FAILED: {str(e)}")
        logger.exception("Traceback:")
        logger.error("="*60)
        
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        
        gc.collect()
        
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/classes', methods=['GET', 'OPTIONS'])
def get_classes():
    if request.method == 'OPTIONS':
        return '', 204
    
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

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Not found'}), 404

@app.errorhandler(413)
def too_large(error):
    return jsonify({'status': 'error', 'message': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.exception("Internal error")
    return jsonify({'status': 'error', 'message': 'Internal error'}), 500

# Startup
logger.info("Starting FarmAI Backend...")
with app.app_context():
    load_model_and_classes()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    
    logger.info("="*60)
    logger.info(f"FarmAI Backend Starting")
    logger.info(f"Port: {port}")
    logger.info(f"Model: {model_loaded}")
    logger.info(f"CPU Optimized: True")
    logger.info("="*60)
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)