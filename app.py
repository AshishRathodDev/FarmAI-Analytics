"""
FarmAI Disease Detection API
Production-grade Flask REST API for crop disease prediction
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
import logging

# ==================== CONFIGURATION ====================

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = 'model.h5'
IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ==================== DISEASE LABELS ====================

# PlantVillage dataset labels (38 classes)
LABEL_MAP = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

# ==================== TREATMENT RECOMMENDATIONS ====================

TREATMENT_MAP = {
    'Apple___Apple_scab': 'Apply fungicide (e.g., Captan, Mancozeb) every 7-10 days. Remove fallen leaves.',
    'Apple___Black_rot': 'Prune infected branches. Apply fungicide during wet seasons. Improve air circulation.',
    'Apple___Cedar_apple_rust': 'Remove nearby cedar trees if possible. Apply fungicide in spring.',
    'Corn_(maize)___Common_rust_': 'Plant resistant varieties. Apply fungicide if severe. Ensure proper spacing.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant hybrids. Apply fungicide at first symptoms. Practice crop rotation.',
    'Grape___Black_rot': 'Remove mummified berries. Apply fungicide (Mancozeb, Captan) regularly.',
    'Grape___Esca_(Black_Measles)': 'No cure available. Remove severely infected vines. Ensure proper nutrition.',
    'Potato___Early_blight': 'Apply fungicide (Chlorothalonil, Mancozeb). Remove lower infected leaves. Mulch soil.',
    'Potato___Late_blight': 'Apply fungicide immediately (Metalaxyl, Mancozeb). Improve drainage. Destroy infected plants.',
    'Tomato___Bacterial_spot': 'Use copper-based bactericides. Avoid overhead watering. Remove infected plants.',
    'Tomato___Early_blight': 'Apply fungicide (Chlorothalonil). Remove infected leaves. Ensure proper spacing.',
    'Tomato___Late_blight': 'Apply fungicide urgently (Mancozeb, Metalaxyl). Destroy infected plants immediately.',
    'Tomato___Leaf_Mold': 'Improve ventilation. Reduce humidity. Apply fungicide (Chlorothalonil).',
    'Tomato___Septoria_leaf_spot': 'Remove infected leaves. Apply fungicide. Avoid overhead irrigation.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Use miticides or neem oil. Increase humidity. Introduce predatory mites.',
    'Tomato___Target_Spot': 'Apply fungicide. Improve air circulation. Practice crop rotation.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'No cure. Control whiteflies. Remove infected plants. Use resistant varieties.',
    'Tomato___Tomato_mosaic_virus': 'No cure. Remove infected plants. Disinfect tools. Plant resistant varieties.',
    'Orange___Haunglongbing_(Citrus_greening)': 'No cure. Remove infected trees. Control Asian citrus psyllid. Use certified clean nursery stock.',
    'Peach___Bacterial_spot': 'Use copper sprays. Prune to improve air flow. Plant resistant varieties.',
    'Pepper,_bell___Bacterial_spot': 'Apply copper bactericide. Remove infected plants. Avoid overhead watering.',
    'Squash___Powdery_mildew': 'Apply fungicide (Sulfur, Potassium bicarbonate). Ensure good air circulation.',
    'Strawberry___Leaf_scorch': 'Remove infected leaves. Apply fungicide. Ensure proper plant spacing.',
    'Cherry_(including_sour)___Powdery_mildew': 'Apply sulfur or fungicide. Prune for better air flow.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Use resistant hybrids. Apply fungicide if needed. Practice crop rotation.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply fungicide. Remove infected leaves. Improve vineyard sanitation.',
    'Soybean___healthy': 'Plant is healthy! Continue regular monitoring and preventive care.',
    'healthy': 'Plant is healthy! Continue with regular care and monitoring.'
}

# Default treatment for healthy plants or unknown diseases
DEFAULT_HEALTHY_TREATMENT = 'Plant appears healthy! Continue regular watering, proper fertilization, and monitor for early disease signs.'
DEFAULT_TREATMENT = 'Consult local agricultural extension office for specific treatment. Practice good sanitation and crop rotation.'

# ==================== GLOBAL MODEL VARIABLE ====================

model = None

# ==================== HELPER FUNCTIONS ====================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_disease_name(disease_label):
    """Format disease label for display"""
    # Replace underscores with spaces and format
    formatted = disease_label.replace('___', ': ').replace('_', ' ')
    return formatted

def get_treatment_recommendation(disease_label):
    """Get treatment recommendation for disease"""
    # Check if disease has specific treatment
    if disease_label in TREATMENT_MAP:
        return TREATMENT_MAP[disease_label]
    
    # Check if healthy
    if 'healthy' in disease_label.lower():
        return DEFAULT_HEALTHY_TREATMENT
    
    # Default treatment
    return DEFAULT_TREATMENT

def preprocess_image(image_bytes):
    """
    Preprocess image for model input
    
    Args:
        image_bytes: Raw image bytes from upload
        
    Returns:
        Preprocessed numpy array ready for prediction
    """
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (handle PNG with alpha channel)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(IMAGE_SIZE)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to [0, 1] (for MobileNet/EfficientNet)
        image_array = image_array.astype('float32') / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")

def load_model():
    """Load TensorFlow model at startup"""
    global model
    
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = keras.models.load_model(MODEL_PATH)
        logger.info(f" Model loaded successfully!")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        
        return True
    
    except Exception as e:
        logger.error(f" Failed to load model: {str(e)}")
        return False

# ==================== API ENDPOINTS ====================

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'FarmAI Disease Detection API is running',
        'model_loaded': model is not None,
        'version': '1.0.0',
        'endpoints': {
            'predict': '/api/predict (POST)',
            'health': '/health (GET)'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check for load balancers"""
    if model is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Expects multipart/form-data with 'file' key containing image
    
    Returns JSON with:
        - disease: Predicted disease name
        - confidence: Model confidence score (0-1)
        - recommendation: Treatment suggestion
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded. Please restart the server.'
            }), 500
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided. Please upload an image with key "file".'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected.'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Read file bytes
        file_bytes = file.read()
        
        # Check file size
        if len(file_bytes) > MAX_FILE_SIZE:
            return jsonify({
                'status': 'error',
                'message': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB'
            }), 400
        
        # Preprocess image
        try:
            processed_image = preprocess_image(file_bytes)
        except ValueError as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 400
        
        # Make prediction
        try:
            predictions = model.predict(processed_image, verbose=0)
            
            # Get predicted class and confidence
            predicted_class_idx = int(np.argmax(predictions[0]))
            confidence_score = float(np.max(predictions[0]))
            
            # Get disease label
            disease_label = LABEL_MAP.get(predicted_class_idx, f'Unknown_Disease_{predicted_class_idx}')
            
            # Format disease name
            disease_name = format_disease_name(disease_label)
            
            # Get treatment recommendation
            recommendation = get_treatment_recommendation(disease_label)
            
            # Get top 3 predictions for additional info
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [
                {
                    'disease': format_disease_name(LABEL_MAP.get(int(idx), f'Unknown_{idx}')),
                    'confidence': float(predictions[0][idx])
                }
                for idx in top_3_indices
            ]
            
            # Log prediction
            logger.info(f"Prediction: {disease_name} (confidence: {confidence_score:.4f})")
            
            # Return success response
            return jsonify({
                'status': 'success',
                'disease': disease_name,
                'confidence': round(confidence_score, 4),
                'confidence_percentage': round(confidence_score * 100, 2),
                'recommendation': recommendation,
                'top_predictions': top_3_predictions,
                'model_version': '1.0'
            }), 200
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Prediction failed. Please try again with a different image.',
                'error_details': str(e)
            }), 500
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred.',
            'error_details': str(e)
        }), 500

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found. Available endpoints: /, /health, /api/predict'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Method not allowed. Use POST for /api/predict'
    }), 405

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return jsonify({
        'status': 'error',
        'message': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error. Please try again later.'
    }), 500

# ==================== APPLICATION STARTUP ====================

def initialize_app():
    """Initialize application and load model"""
    logger.info("=" * 50)
    logger.info(" FarmAI Disease Detection API Starting...")
    logger.info("=" * 50)
    
    # Load model
    model_loaded = load_model()
    
    if not model_loaded:
        logger.warning("  API starting without model. Predictions will fail.")
    
    logger.info("=" * 50)
    logger.info(" API Ready!")
    logger.info(" Available endpoints:")
    logger.info(" GET  /           - API info")
    logger.info(" GET  /health     - Health check")
    logger.info(" POST /api/predict - Disease prediction")
    logger.info("=" * 50)

# ==================== MAIN ENTRY POINT ====================

if __name__ == '__main__':
    # Initialize app
    initialize_app()
    
    # Run Flask app
    # For production, use Gunicorn: gunicorn -w 4 -b 0.0.0.0:5000 app:app
    app.run(
        host='0.0.0.0',
        port=5050,
        debug=False  # Set to False in production
    )
    
    
    