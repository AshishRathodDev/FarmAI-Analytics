"""
FarmAI Backend API - Production Version
Refactored with proper error handling, validation, and thread safety.
This file serves as the main entry point for the Flask application.
"""

import io
import logging
import time
from pathlib import Path
import os
import random # For demo predictions

from flask import Flask, jsonify, request
from flask_cors import CORS

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.ml.model_loader import model_loader
from app.core.database import init_database, get_db_manager, get_disease_id_by_name
from app.api.middleware.validation import validate_image_upload, validate_chat_request
from app.api.middleware.rate_limit import rate_limit
from app.api.middleware.auth import require_api_key # For securing monitoring endpoints
from app.api.utils.sanitization import sanitize_string # For chat input sanitization

# Import blueprints
from app.api.routes.prediction import prediction_bp
from app.api.routes.chat import chat_bp
from app.api.routes.analytics import analytics_bp
from app.api.routes.monitoring import monitoring_bp


# Setup logging based on settings
setup_logging(settings.LOG_FILE, settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=settings.CORS_ORIGINS)

# Configuration for file uploads
app.config['MAX_CONTENT_LENGTH'] = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Register blueprints
app.register_blueprint(prediction_bp, url_prefix='/api')
app.register_blueprint(chat_bp, url_prefix='/api')
app.register_blueprint(analytics_bp, url_prefix='/api')
app.register_blueprint(monitoring_bp, url_prefix='/api/monitoring')


# Global resources (initialized in initialize_resources)
global_chatbot_instance = None
global_analytics_engine_instance = None
_init_lock = threading.Lock() # Thread-safe initialization lock

def initialize_resources():
    """Initialize all application resources (thread-safe)."""
    global global_chatbot_instance, global_analytics_engine_instance
    
    with _init_lock:
        # Check if already initialized in this process/worker
        if global_chatbot_instance is not None and model_loader.is_loaded():
            logger.info("Resources already initialized in this process.")
            return

        logger.info("Initializing application resources...")
        
        # Initialize database
        try:
            init_database()
        except Exception as e:
            logger.exception("Database initialization failed: %s", e)
            
        # Load ML model
        success = model_loader.load(settings.MODEL_PATH, settings.CLASS_INDICES_PATH)
        if success:
            logger.info("ML model loaded successfully.")
        else:
            logger.warning("ML model failed to load. Prediction service will run in demo mode.")
        
        # Initialize chatbot
        if settings.GOOGLE_API_KEY:
            try:
                from app.services.chatbot import FarmAIChatbot
                global_chatbot_instance = FarmAIChatbot(settings.GOOGLE_API_KEY, settings.CHATBOT_MODEL)
                logger.info("Chatbot service initialized.")
            except Exception as e:
                logger.error("Chatbot initialization failed: %s", e)
                global_chatbot_instance = None
        else:
            logger.warning("GOOGLE_API_KEY not set. Chatbot service will be unavailable.")
        
        # Initialize analytics
        try:
            from app.services.analytics import AnalyticsEngine
            db_manager = get_db_manager() # Get manager from core.database
            if db_manager:
                global_analytics_engine_instance = AnalyticsEngine(db_manager)
                logger.info("Analytics engine initialized.")
            else:
                logger.warning("Database manager not available, Analytics engine not initialized.")
        except Exception as e:
            logger.error("Analytics initialization failed: %s", e)
            global_analytics_engine_instance = None
        
        logger.info("Application resource initialization complete.")


# Helper functions (moved to global scope for easy access by routes)
def get_farmer_id():
    """Extract or generate farmer ID from request headers."""
    # In a real app, this would come from user authentication (e.g., JWT token, session cookie)
    return request.headers.get('X-Farmer-ID', f"anon_{int(time.time())}")

def get_treatment_recommendation(disease_name: str) -> str:
    """Get treatment recommendation for disease."""
    # This data can be moved to a DB table or a separate JSON/YAML file
    treatment_db = {
        "Apple: Apple Scab": (
            "Apply fungicide (e.g., Captan, Mancozeb) every 7-10 days. "
            "Remove fallen leaves and prune infected branches."
        ),
        "Apple: Black Rot": (
            "Prune infected branches. Apply fungicide during wet seasons. Improve air circulation."
        ),
        "Apple: Cedar Apple Rust": (
            "Remove nearby cedar trees if possible. Apply fungicide in spring (e.g., Myclobutanil)."
        ),
        "Corn (Maize): Common Rust": (
            "Plant resistant varieties. Apply fungicide (e.g., Azoxystrobin) if severe. Ensure proper spacing."
        ),
        "Corn (Maize): Northern Leaf Blight": (
            "Use resistant hybrids. Apply fungicide at first symptoms. Practice crop rotation."
        ),
        "Grape: Black Rot": (
            "Remove mummified berries. Apply fungicide (e.g., Mancozeb, Captan) regularly."
        ),
        "Grape: Esca (Black Measles)": (
            "No chemical cure available. Remove severely infected vines. Ensure proper nutrition and vineyard hygiene."
        ),
        "Potato: Early Blight": (
            "Apply fungicide (e.g., Chlorothalonil, Mancozeb). Remove lower infected leaves. Mulch soil."
        ),
        "Potato: Late Blight": (
            "Apply fungicide immediately (e.g., Metalaxyl, Mancozeb). Improve drainage. Destroy infected plants."
        ),
        "Tomato: Bacterial Spot": (
            "Use copper-based bactericides. Avoid overhead watering. Remove and destroy infected plants."
        ),
        "Tomato: Early Blight": (
            "Apply fungicide (e.g., Chlorothalonil). Remove infected leaves. Ensure proper spacing."
        ),
        "Tomato: Late Blight": (
            "Apply fungicide urgently (e.g., Mancozeb, Metalaxyl). Destroy infected plants immediately."
        ),
        "Tomato: Leaf Mold": (
            "Improve greenhouse ventilation. Reduce humidity. Apply fungicide (e.g., Chlorothalonil)."
        ),
        "Tomato: Septoria Leaf Spot": (
            "Remove and destroy infected leaves. Apply fungicide. Avoid overhead irrigation. Practice crop rotation."
        ),
        "Tomato: Spider Mites Two-Spotted Spider Mite": (
            "Use miticides or neem oil. Increase humidity. Introduce predatory mites."
        ),
        "Tomato: Target Spot": (
            "Apply fungicide. Improve air circulation. Practice crop rotation."
        ),
        "Tomato: Tomato Yellow Leaf Curl Virus": (
            "No chemical cure. Control whiteflies. Remove infected plants. Use resistant varieties."
        ),
        "Tomato: Tomato Mosaic Virus": (
            "No cure. Remove infected plants. Disinfect tools. Plant resistant varieties."
        ),
        "Orange: Haunglongbing (Citrus Greening)": (
            "No cure. Remove infected trees. Control Asian citrus psyllid. Use certified clean nursery stock."
        ),
        "Peach: Bacterial Spot": (
            "Use copper sprays. Prune to improve air flow. Plant resistant varieties."
        ),
        "Pepper, Bell: Bacterial Spot": (
            "Apply copper bactericide. Remove infected plants. Avoid overhead watering."
        ),
        "Squash: Powdery Mildew": (
            "Apply fungicide (e.g., Sulfur, Potassium bicarbonate). Ensure good air circulation."
        ),
        "Strawberry: Leaf Scorch": (
            "Remove infected leaves. Apply fungicide. Ensure proper plant spacing."
        ),
        "Cherry (Including Sour): Powdery Mildew": (
            "Apply sulfur or fungicide. Prune for better air flow."
        ),
        # Default healthy treatments
        "healthy": "Plant appears healthy! Continue regular watering, proper fertilization, and monitor for early disease signs.",
    }

    formatted_disease_name = disease_name.replace('___', ': ').replace('_', ' ').title()

    # Try to find specific treatment
    for key, treatment in treatment_db.items():
        if key.lower() == formatted_disease_name.lower():
            return treatment

    # Fallback for generic healthy or unknown diseases
    if "healthy" in formatted_disease_name.lower():
        return treatment_db.get("healthy", "Plant appears healthy! Continue regular care.")
    
    return (
        "General Disease Management:\n"
        "1. Remove infected plant parts and destroy.\n"
        "2. Improve air circulation.\n"
        "3. Apply appropriate fungicide/bactericide.\n"
        "4. Consult local agricultural extension.\n"
        "5. Practice crop rotation."
    )

def determine_severity(confidence: float, disease_name: str) -> str:
    """Determine disease severity level based on confidence and disease type."""
    
    if "healthy" in disease_name.lower():
        return "Low"

    high_severity_keywords = ["blight", "rot", "wilt", "virus", "mosaic", "scab"]
    if any(keyword in disease_name.lower() for keyword in high_severity_keywords):
        return "High" if confidence >= 0.80 else "Medium"

    medium_keywords = ["spot", "mold", "rust"]
    if any(keyword in disease_name.lower() for keyword in medium_keywords):
        return "Medium"

    return "Medium" if confidence >= 0.85 else "Low"


# Error handlers (registered at global app level)
@app.errorhandler(404)
def not_found(error):
    logger.warning("404 Not Found: %s", request.url)
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.errorhandler(405) # Method Not Allowed is also handled by Flask's routing
def method_not_allowed(error):
    logger.warning("405 Method Not Allowed: %s for %s", request.method, request.url)
    return jsonify({"status": "error", "message": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_server_error_handler(error):
    logger.exception("Internal server error: %s", error)
    return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.errorhandler(413) # Payload Too Large is handled by Flask's MAX_CONTENT_LENGTH
def request_entity_too_large(error):
    logger.warning("413 Request entity too large: %s", error)
    return jsonify({"status": "error", "message": f"File too large. Maximum: {settings.MAX_UPLOAD_SIZE_MB}MB"}), 413


# Initialize resources when Flask app context is available (production-ready)
# This will run once when the application starts, before any requests are processed.
with app.app_context():
    initialize_resources()


# --- Main Entry Point ---
if __name__ == '__main__':
    logger.info("==================================================")
    logger.info(" FarmAI Backend Service Starting...")
    logger.info("==================================================")
    
    logger.info("API will be serving on host %s port %s", settings.API_HOST, settings.API_PORT)
    
    # Run Flask app (debug=False for production readiness)
    app.run(
        host=settings.API_HOST,
        port=settings.API_PORT,
        debug=settings.DEBUG # Should be False in production
    )