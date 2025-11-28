"""
FarmAI Flask API - PRODUCTION READY (WITH RETRY LOGIC)
"""


import os
import sys



import json
import logging
import gc
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ============================================
# FORCE CPU MODE
# ============================================
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ============================================
# LOGGING
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================
# FLASK APP
# ============================================
app = Flask(__name__)

# CORS Configuration
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"],
     expose_headers=["Content-Type"],
     supports_credentials=False,
     max_age=3600
)

@app.after_request
def after_request(response):
    """Force CORS headers"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# ============================================
# CONFIG
# ============================================
UPLOAD_FOLDER = '/tmp/uploads'
MODEL_DIR = '/tmp/models'
CACHE_DIR = '/tmp/hf_cache'
MODEL_FILENAME = 'crop_disease_classifier_final.keras'
CLASS_INDICES_FILENAME = 'class_indices.json'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, CLASS_INDICES_FILENAME)
HUGGINGFACE_REPO = "rathodashish10/farmai-models"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# Create directories
for directory in [UPLOAD_FOLDER, MODEL_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f" Directory created/verified: {directory}")

# Global state
model = None
class_names = []
model_loaded = False
model_loading_error = None

# ============================================
# HELPER FUNCTIONS
# ============================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_disease_name(class_name):
    return class_name.replace('___', ': ').replace('_', ' ').title()

# ============================================
# MODEL LOADING WITH RETRY
# ============================================
def download_with_retry(repo_id, filename, local_dir, max_retries=3):
    """Download file from Hugging Face with retry logic"""
    from huggingface_hub import hf_hub_download
    
    for attempt in range(max_retries):
        try:
            logger.info(f" Download attempt {attempt + 1}/{max_retries}: {filename}")
            
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                cache_dir=CACHE_DIR,
                resume_download=True,
                force_download=False
            )
            
            logger.info(f" Downloaded successfully: {filename}")
            return file_path
            
        except Exception as e:
            logger.error(f" Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                logger.info(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed to download {filename} after {max_retries} attempts: {e}")

def load_model():
    """Download and load model with comprehensive error handling"""
    global model, class_names, model_loaded, model_loading_error
    
    try:
        logger.info("=" * 70)
        logger.info(" MODEL LOADING PROCESS STARTED")
        logger.info("=" * 70)
        
        # Check directories
        logger.info(f" Upload folder: {UPLOAD_FOLDER}")
        logger.info(f" Model folder: {MODEL_DIR}")
        logger.info(f" Cache folder: {CACHE_DIR}")
        
        # Import required libraries
        logger.info(" Importing TensorFlow and Keras...")
        import keras
        import numpy as np
        from PIL import Image
        logger.info(" Libraries imported successfully")
        
        # Download class indices first (smaller file)
        logger.info(f" Loading class indices from Hugging Face...")
        if not os.path.exists(CLASS_INDICES_PATH):
            download_with_retry(
                repo_id=HUGGINGFACE_REPO,
                filename=CLASS_INDICES_FILENAME,
                local_dir=MODEL_DIR
            )
        else:
            logger.info(f" Class indices already exist: {CLASS_INDICES_PATH}")
        
        # Load class names
        logger.info(" Reading class indices...")
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_data = json.load(f)
            class_names = list(class_data.keys())
        
        logger.info(f" Loaded {len(class_names)} disease classes:")
        for i, name in enumerate(class_names[:5], 1):
            logger.info(f"   {i}. {format_disease_name(name)}")
        if len(class_names) > 5:
            logger.info(f"   ... and {len(class_names) - 5} more")
        
        # Download model
        logger.info(f" Loading Keras model from Hugging Face...")
        if not os.path.exists(MODEL_PATH):
            download_with_retry(
                repo_id=HUGGINGFACE_REPO,
                filename=MODEL_FILENAME,
                local_dir=MODEL_DIR
            )
        else:
            logger.info(f" Model file already exists: {MODEL_PATH}")
            # Verify file size
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            logger.info(f" Model file size: {file_size:.2f} MB")
        
        # Load model into memory
        logger.info(" Loading model into memory...")
        model = keras.models.load_model(MODEL_PATH, compile=False)
        
        logger.info("=" * 70)
        logger.info(" MODEL LOADED SUCCESSFULLY!")
        logger.info(f" Model input shape: {model.input_shape}")
        logger.info(f" Model output shape: {model.output_shape}")
        logger.info(f" Total classes: {len(class_names)}")
        logger.info("=" * 70)
        
        model_loaded = True
        model_loading_error = None
        return True
        
    except Exception as e:
        error_msg = f"Model loading failed: {str(e)}"
        logger.error("=" * 70)
        logger.error(f" {error_msg}")
        logger.error("=" * 70)
        logger.error("Stack trace:", exc_info=True)
        
        model_loaded = False
        model_loading_error = error_msg
        return False

# ============================================
# ROUTES
# ============================================

@app.route('/', methods=['GET', 'OPTIONS'])
def home():
    if request.method == 'OPTIONS':
        return '', 200
    
    return jsonify({
        'service': 'FarmAI Disease Detection API',
        'version': '3.0.0',
        'status': 'running',
        'model_loaded': model_loaded,
        'model_error': model_loading_error,
        'endpoints': {
            'health': '/health',
            'predict': '/api/predict (POST)',
            'classes': '/api/classes'
        }
    }), 200

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 200
    
    response = {
        'status': 'healthy',
        'model_loaded': model_loaded,
        'classes_count': len(class_names),
        'cpu_mode': True
    }
    
    if model_loading_error:
        response['model_error'] = model_loading_error
        response['suggestion'] = 'Check Cloud Run logs for details'
    
    return jsonify(response), 200

@app.route('/api/classes', methods=['GET', 'OPTIONS'])
def get_classes():
    if request.method == 'OPTIONS':
        return '', 200
    
    if not model_loaded:
        return jsonify({
            'status': 'error',
            'error': 'Model not loaded yet',
            'model_error': model_loading_error
        }), 503
    
    formatted = [format_disease_name(name) for name in class_names]
    return jsonify({
        'status': 'success',
        'count': len(formatted),
        'classes': formatted
    }), 200

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        logger.info(" OPTIONS preflight request handled")
        return '', 200
    
    logger.info("=" * 70)
    logger.info(" PREDICTION REQUEST RECEIVED")
    logger.info(f"   Method: {request.method}")
    logger.info(f"   Content-Type: {request.content_type}")
    logger.info(f"   Files: {list(request.files.keys())}")
    logger.info("=" * 70)
    
    filepath = None
    start_time = time.time()
    
    try:
        # Check model status
        if not model_loaded:
            error_msg = model_loading_error or 'Model is still loading'
            logger.warning(f" Prediction blocked: {error_msg}")
            return jsonify({
                'status': 'error',
                'error': 'Model is loading. Please wait and try again.',
                'details': error_msg
            }), 503
        
        # Validate file
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'error': 'Empty filename'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        file_size = os.path.getsize(filepath) / 1024
        logger.info(f" Image saved: {filename} ({file_size:.2f} KB)")
        
        # Preprocess
        import numpy as np
        from PIL import Image
        
        logger.info(" Preprocessing...")
        with Image.open(filepath) as img:
            img = img.convert('RGB').resize((160, 160))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        logger.info(" Running inference...")
        inference_start = time.time()
        predictions = model.predict(img_array, verbose=0)[0]
        inference_time = time.time() - inference_start
        
        # Results
        top_idx = np.argmax(predictions)
        confidence = float(predictions[top_idx])
        disease = format_disease_name(class_names[top_idx])
        
        # Top 3
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3 = [
            {
                'disease': format_disease_name(class_names[idx]),
                'confidence': float(predictions[idx]),
                'confidence_percent': f"{predictions[idx]*100:.2f}%"
            }
            for idx in top_3_indices
        ]
        
        total_time = time.time() - start_time
        logger.info(f" Prediction: {disease} ({confidence*100:.2f}%)")
        logger.info(f"⏱ Time: {inference_time:.2f}s inference, {total_time:.2f}s total")
        
        return jsonify({
            'status': 'success',
            'prediction': disease,
            'confidence': confidence,
            'confidence_percent': f"{confidence*100:.2f}%",
            'top_3': top_3,
            'inference_time': f"{inference_time:.2f}s",
            'total_time': f"{total_time:.2f}s"
        }), 200
        
    except Exception as e:
        logger.error(f" Prediction error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500
        
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        gc.collect()

@app.errorhandler(404)
def not_found(e):
    return jsonify({'status': 'error', 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'status': 'error', 'error': 'Internal server error'}), 500

# ============================================
# STARTUP
# ============================================

with app.app_context():
    logger.info("")
    logger.info("🌾" * 35)
    logger.info("    FARMAI BACKEND - STARTUP")
    logger.info("🌾" * 35)
    logger.info("")
    
    success = load_model()
    
    if success:
        logger.info("")
        logger.info("" * 35)
        logger.info("    BACKEND READY TO SERVE!")
        logger.info("" * 35)
        logger.info("")
    else:
        logger.error("")
        logger.error("" * 35)
        logger.error("    MODEL LOADING FAILED!")
        logger.error("    Check logs above for details")
        logger.error("" * 35)
        logger.error("")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f" Starting Flask on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    
    

    