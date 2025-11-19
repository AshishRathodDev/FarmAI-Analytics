from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
import json
import logging
from pathlib import Path
import random
import time

# --- Configuration (from config.py) ---
try:
    from config import (
        MODEL_PATH,
        DATABASE_PATH,
        IMAGE_SIZE,
        GOOGLE_API_KEY
    )
    # Handle CLASS_INDICES_PATH
    CLASS_INDICES_PATH = Path('models/class_indices.json')
except ImportError as e:
    print(f"⚠️  Config import error: {e}")
    # Fallback defaults
    MODEL_PATH = 'models/crop_disease_classifier_final.h5'
    DATABASE_PATH = 'farmer_analytics.db'
    IMAGE_SIZE = (224, 224)
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    CLASS_INDICES_PATH = Path('models/class_indices.json')

# --- Your Project Imports (from src/ folder) ---
try:
    from src.database_manager import FarmAIDatabaseManager
    from src.crop_classifier import CropDiseaseClassifier
    from src.chatbot_agent import FarmAIChatbot
    from src.analytics_engine import AnalyticsEngine
except ImportError as e:
    print(f"⚠️  Module import error: {e}")
    print("Make sure all src/ modules exist")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# --- Global Objects ---
db_manager = None
classifier = None
chatbot = None
analytics_engine = None
CLASS_NAMES = None

def load_global_resources():
    """Loads database, model, and chatbot objects globally."""
    global db_manager, classifier, chatbot, analytics_engine, CLASS_NAMES

    logger.info("🚀 Loading global resources for FarmAI backend...")

    # Initialize Database Manager
    try:
        if db_manager is None:
            db_manager = FarmAIDatabaseManager(DATABASE_PATH)
            logger.info(f"✅ Database Manager initialized using {DATABASE_PATH}")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")

    # Load Class Names
    try:
        if CLASS_NAMES is None:
            if CLASS_INDICES_PATH.exists():
                with open(CLASS_INDICES_PATH, 'r') as f:
                    class_indices = json.load(f)
                    # Reverse the dict to get class names by index
                    CLASS_NAMES = {int(idx): name for name, idx in class_indices.items()}
                logger.info(f"✅ Loaded {len(CLASS_NAMES)} class names from {CLASS_INDICES_PATH}")
            else:
                logger.warning(f"⚠️  Class indices file not found at {CLASS_INDICES_PATH}")
                # Fallback class names
                CLASS_NAMES = {
                    0: 'Apple___Apple_scab',
                    1: 'Apple___Black_rot',
                    2: 'Apple___Cedar_apple_rust',
                    3: 'Apple___healthy',
                    10: 'Potato___Early_blight',
                    11: 'Potato___Late_blight',
                    12: 'Potato___healthy',
                    13: 'Tomato___Bacterial_spot',
                    14: 'Tomato___Early_blight',
                    15: 'Tomato___Late_blight',
                    16: 'Tomato___Leaf_Mold',
                    17: 'Tomato___Septoria_leaf_spot',
                    18: 'Tomato___healthy'
                }
    except Exception as e:
        logger.error(f"❌ Class names loading failed: {e}")
        CLASS_NAMES = {i: f"Disease_Class_{i}" for i in range(38)}

    # Initialize Crop Classifier
    try:
        if classifier is None:
            classifier = CropDiseaseClassifier(MODEL_PATH)
            logger.info(f"✅ Crop Disease Classifier initialized using {MODEL_PATH}")
            if not classifier.model:
                logger.warning("⚠️  Classifier model not loaded. Predictions will run in DEMO mode.")
    except Exception as e:
        logger.error(f"❌ Classifier initialization failed: {e}")

    # Initialize Chatbot
    try:
        if chatbot is None:
            if GOOGLE_API_KEY:
                chatbot = FarmAIChatbot(GOOGLE_API_KEY)
                logger.info("✅ Chatbot initialized with Google Gemini")
            else:
                logger.warning("⚠️  GOOGLE_API_KEY not found. Chatbot will be limited.")
    except Exception as e:
        logger.error(f"❌ Chatbot initialization failed: {e}")

    # Initialize Analytics Engine
    try:
        if analytics_engine is None and db_manager:
            analytics_engine = AnalyticsEngine(db_manager)
            logger.info("✅ Analytics Engine initialized")
    except Exception as e:
        logger.error(f"❌ Analytics initialization failed: {e}")

# Load resources before first request
@app.before_request
def before_first_request():
    global db_manager, classifier, chatbot, analytics_engine
    if db_manager is None:
        load_global_resources()

# --- Utility Functions ---
def get_farmer_id():
    """Get farmer ID from request header or generate anonymous one"""
    return request.headers.get('X-Farmer-ID', f'anon_{int(time.time())}')

def format_disease_name(disease_label):
    """Format disease label for display"""
    formatted = disease_label.replace('___', ': ').replace('_', ' ')
    return formatted

# --- API Endpoints ---

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        "status": "success",
        "message": "FarmAI Backend API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict (POST)",
            "chat": "/api/chat (POST)",
            "analytics_summary": "/api/analytics/summary",
            "analytics_trends": "/api/analytics/trends",
            "disease_distribution": "/api/analytics/diseases"
        }
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_loaded = bool(classifier and classifier.model)
    chatbot_ready = bool(chatbot)
    db_ready = bool(db_manager)
    
    status = "healthy" if (model_loaded and db_ready) else "degraded"
    
    return jsonify({
        "status": status,
        "message": "FarmAI backend is running",
        "model_loaded": model_loaded,
        "chatbot_initialized": chatbot_ready,
        "database_connected": db_ready,
        "timestamp": time.time()
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict_disease_api():
    """
    Endpoint to predict crop disease from uploaded image
    Accepts: multipart/form-data with 'file' key (NOT 'image')
    """
    # Check both 'file' (from React) and 'image' (legacy)
    file = request.files.get('file') or request.files.get('image')
    
    if not file:
        return jsonify({
            "status": "error",
            "message": "No image file provided. Use 'file' key in form-data."
        }), 400

    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No selected image file"
        }), 400

    try:
        # Read image
        image_data = file.read()
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Make prediction
        if classifier and classifier.model:
            logger.info("🔬 Making real prediction with model...")
            prediction_result = classifier.predict(img)
        else:
            logger.warning("⚠️  Model not loaded. Using DEMO prediction")
            # Simulate prediction
            demo_diseases = [
                'Tomato: Early Blight',
                'Potato: Late Blight', 
                'Apple: Apple Scab',
                'Tomato: Healthy',
                'Potato: Healthy'
            ]
            prediction_result = {
                'disease': random.choice(demo_diseases),
                'confidence': round(random.uniform(0.7, 0.95), 4),
                'confidence_percentage': round(random.uniform(70, 95), 2),
                'treatment': 'This is a DEMO prediction. Please load the model for real results.',
                'severity': random.choice(['High', 'Medium', 'Low']),
                'prediction_time': round(random.uniform(0.1, 0.5), 3),
                'model_version': 'DEMO_v1.0',
                'is_confident': True
            }

        # Log to database
        try:
            farmer_id = get_farmer_id()
            if db_manager:
                # Ensure farmer exists
                db_manager.add_farmer(farmer_id, name="Anonymous")
                
                # Save prediction
                db_manager.save_prediction(
                    farmer_id=farmer_id,
                    predicted_disease_id=1,  # Default ID
                    confidence=prediction_result['confidence'],
                    model_version=prediction_result.get('model_version', '1.0'),
                    prediction_time=prediction_result['prediction_time'],
                    image_file=file.filename
                )
                logger.info(f"✅ Prediction logged for farmer {farmer_id}")
        except Exception as db_error:
            logger.error(f"Database logging failed: {db_error}")
        
        # Return success response
        return jsonify({
            "status": "success",
            **prediction_result
        }), 200

    except Exception as e:
        logger.error(f"❌ Prediction API error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_ai_api():
    """
    AI chatbot endpoint
    Accepts JSON: { "message": "your question", "crop": "Tomato", "language": "English" }
    """
    data = request.get_json()
    
    # Handle both 'message' (from React) and 'query' (legacy)
    farmer_query = data.get('message') or data.get('query')
    crop_type = data.get('crop', 'General')
    disease_context = data.get('disease_context')
    language = data.get('language', 'English')

    if not farmer_query:
        return jsonify({
            "status": "error",
            "message": "No message/query provided"
        }), 400

    if not chatbot:
        return jsonify({
            "status": "error",
            "message": "Chatbot service unavailable. Please configure GOOGLE_API_KEY"
        }), 503

    try:
        logger.info(f"💬 Chat request: {farmer_query[:50]}...")
        
        response_text, response_time = chatbot.generate_response(
            farmer_query=farmer_query,
            crop_type=crop_type,
            disease_name=disease_context,
            language=language
        )

        # Log to database
        try:
            farmer_id = get_farmer_id()
            if db_manager:
                db_manager.log_chatbot_interaction(
                    farmer_id=farmer_id,
                    query=farmer_query,
                    response=response_text,
                    language=language,
                    response_time=response_time
                )
        except Exception as db_error:
            logger.error(f"Database logging failed: {db_error}")

        return jsonify({
            "status": "success",
            "response": response_text,
            "responseTime": round(response_time, 2)
        }), 200

    except Exception as e:
        logger.error(f"❌ Chat API error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Chat failed: {str(e)}"
        }), 500

@app.route('/api/analytics/summary', methods=['GET'])
def get_analytics_summary_api():
    """Get analytics summary metrics"""
    try:
        if not analytics_engine:
            return jsonify({
                "status": "error",
                "message": "Analytics engine not initialized"
            }), 503
        
        summary = analytics_engine.get_dashboard_metrics()
        return jsonify({
            "status": "success",
            **summary
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Analytics summary error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Could not fetch analytics: {str(e)}"
        }), 500

@app.route('/api/analytics/trends', methods=['GET'])
def get_analytics_trends_api():
    """Get time-series trend data"""
    days = request.args.get('days', 30, type=int)
    
    try:
        if not analytics_engine:
            return jsonify({
                "status": "error",
                "message": "Analytics engine not initialized"
            }), 503
        
        trends = analytics_engine.get_trend_analysis(days)
        return jsonify({
            "status": "success",
            **trends
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Analytics trends error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Could not fetch trends: {str(e)}"
        }), 500

@app.route('/api/analytics/diseases', methods=['GET'])
def get_disease_distribution_api():
    """Get disease distribution data"""
    limit = request.args.get('limit', 10, type=int)
    
    try:
        if not analytics_engine:
            return jsonify({
                "status": "error",
                "message": "Analytics engine not initialized"
            }), 503
        
        disease_dist_df = analytics_engine.get_disease_distribution(limit)
        return jsonify({
            "status": "success",
            "data": disease_dist_df.to_dict(orient='records')
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Disease distribution error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Could not fetch disease data: {str(e)}"
        }), 500

# --- Error Handlers ---

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "available_endpoints": ["/api/health", "/api/predict", "/api/chat"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500

# --- Main Entry Point ---
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 FarmAI API Server Starting...")
    print("=" * 60)
    
    # Load resources
    load_global_resources()
    
    print(f"📡 Backend running at http://localhost:5050")
    print(f"🏥 Health check: http://localhost:5050/api/health")
    print("=" * 60)
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5050,
        debug=True  # Set False for production
    )