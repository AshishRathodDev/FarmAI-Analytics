"""
FarmAI Backend API (Flask) - Stable production/dev-ready version

Notes:
- Uses Path objects from config where possible.
- Safe initialization if model/chatbot or DB modules are missing.
- Improved logging to a single log file (configured in config.py).
"""

import io
import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

# ----- Config import (use fallback defaults if config missing) -----
try:
    from config import (
        MODEL_PATH,
        DATABASE_PATH,
        IMAGE_SIZE,
        GOOGLE_API_KEY,
        CLASS_INDICES_PATH,
        LOG_FILE,
    )
    # Ensure Path types
    MODEL_PATH = Path(MODEL_PATH)
    DATABASE_PATH = Path(DATABASE_PATH)
    CLASS_INDICES_PATH = Path(CLASS_INDICES_PATH)
    LOG_FILE = Path(LOG_FILE)
except Exception as e:
    # If config import fails, use conservative defaults
    print("Config import failed:", e)
    MODEL_PATH = Path("models/crop_disease_classifier_final.h5")
    DATABASE_PATH = Path("farmer_analytics.db")
    IMAGE_SIZE = (224, 224)
    GOOGLE_API_KEY = None
    CLASS_INDICES_PATH = Path("models/class_indices.json")
    LOG_FILE = Path("logs/app.log")

# ----- Project module imports (safe) -----
try:
    from src.database_manager import FarmAIDatabaseManager
except Exception as e:
    FarmAIDatabaseManager = None
    print("Warning: database_manager import failed:", e)

try:
    from src.crop_classifier import CropDiseaseClassifier
except Exception as e:
    CropDiseaseClassifier = None
    print("Warning: crop_classifier import failed:", e)

try:
    from src.chatbot_agent import FarmAIChatbot
except Exception as e:
    FarmAIChatbot = None
    print("Warning: chatbot_agent import failed:", e)

try:
    from src.analytics_engine import AnalyticsEngine
except Exception as e:
    AnalyticsEngine = None
    print("Warning: analytics_engine import failed:", e)

# ----- Logging setup -----
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("farmai.api")

# ----- Flask app -----
app = Flask(__name__)
CORS(app)

# ----- Global singletons -----
db_manager: Optional[object] = None
classifier: Optional[object] = None
chatbot: Optional[object] = None
analytics_engine: Optional[object] = None
CLASS_NAMES = None


def load_global_resources():
    """Initialize and load DB, classifier, chatbot and analytics engine."""
    global db_manager, classifier, chatbot, analytics_engine, CLASS_NAMES

    logger.info("Loading global resources...")

    # Database Manager
    if db_manager is None and FarmAIDatabaseManager is not None:
        try:
            db_manager = FarmAIDatabaseManager(str(DATABASE_PATH))
            logger.info("Database manager initialized.")
        except Exception as e:
            logger.exception("Database initialization failed: %s", e)
            db_manager = None

    # Class indices / class names
        try:
            if CLASS_NAMES is None:
                CLASS_INDICES_PATH = Path(CLASS_INDICES_PATH)  # ensure Path
                if CLASS_INDICES_PATH.exists():
                    with open(CLASS_INDICES_PATH, "r") as f:
                        class_indices = json.load(f)
                        # Normalize mapping: allow either {"0": "name",...} or {"name": 0, ...}
                        if all(str(k).isdigit() for k in class_indices.keys()):
                            CLASS_NAMES = {int(k): v for k, v in class_indices.items()}
                        else:
                        # reverse mapping assumed: name -> index
                            CLASS_NAMES = {int(v): k for k, v in class_indices.items()}
                    logger.info("✅ Loaded class indices (%d classes) from %s", len(CLASS_NAMES), CLASS_INDICES_PATH)
                else:
                    logger.warning("⚠️ Class indices file not found: %s", CLASS_INDICES_PATH)
                    CLASS_NAMES = {}
        except Exception:
            logger.exception("❌ Failed to load class indices, continuing with fallback.")
            CLASS_NAMES = {}
    
    # Classifier

    # --- Classifier init (robust) ---
    if classifier is None and 'CropDiseaseClassifier' in globals():
        try:
            model_path_str = str(MODEL_PATH) if isinstance(MODEL_PATH, (Path,)) else MODEL_PATH
            logger.info("Attempting to load classifier from: %s", model_path_str)

            classifier = CropDiseaseClassifier(model_path_str)

            # Try to get model info if available
            if classifier and getattr(classifier, "model", None):
                info = {}
                try:
                    info = classifier.get_model_info() if hasattr(classifier, "get_model_info") else {}
                except Exception:
                    logger.exception("Could not call get_model_info() on classifier")
                logger.info("Classifier loaded successfully. model info: %s", info)
            else:
                logger.warning("Classifier initialized but model is NOT loaded (DEMO mode).")
                logger.warning("Checked model path: %s", model_path_str)

        except Exception as exc:
            logger.exception("Failed to initialize classifier: %s", exc)
            classifier = None
    
    
    # Chatbot
    if chatbot is None and FarmAIChatbot is not None:
        try:
            if GOOGLE_API_KEY:
                chatbot = FarmAIChatbot(GOOGLE_API_KEY)
                logger.info("Chatbot initialized.")
            else:
                logger.warning("GOOGLE_API_KEY not set; chatbot will be unavailable.")
                chatbot = None
        except Exception:
            logger.exception("Failed to initialize chatbot.")
            chatbot = None

    # Analytics Engine
    if analytics_engine is None and AnalyticsEngine is not None and db_manager is not None:
        try:
            analytics_engine = AnalyticsEngine(db_manager)
            logger.info("Analytics engine initialized.")
        except Exception:
            logger.exception("Failed to initialize analytics engine.")
            analytics_engine = None


@app.before_request
def ensure_resources_loaded():
    """Make sure resources are loaded before handling requests."""
    global db_manager
    if db_manager is None:
        load_global_resources()


# ----- Helpers -----
def get_farmer_id() -> str:
    """Get farmer id from header or generate an anonymous one."""
    header = request.headers.get("X-Farmer-ID")
    if header:
        return header
    return f"anon_{int(time.time())}"


def format_disease_name(disease_label: str) -> str:
    """Human-friendly disease name."""
    return disease_label.replace("___", ": ").replace("_", " ")


# ----- Endpoints -----
@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "success",
            "message": "FarmAI Backend API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/api/health",
                "predict": "/api/predict (POST form-data 'file')",
                "chat": "/api/chat (POST JSON {message})",
                "analytics_summary": "/api/analytics/summary",
            },
        }
    ), 200


@app.route("/api/health", methods=["GET"])
def health_check():
    model_loaded = bool(classifier and getattr(classifier, "model", None))
    chatbot_ready = bool(chatbot)
    db_ready = bool(db_manager)
    status = "healthy" if (model_loaded and db_ready) else "degraded"
    return jsonify(
        {
            "status": status,
            "model_loaded": model_loaded,
            "chatbot_initialized": chatbot_ready,
            "database_connected": db_ready,
            "timestamp": time.time(),
        }
    ), 200


@app.route("/api/predict", methods=["POST"])
def predict_disease_api():
    """
    Accepts multipart/form-data with key 'file' (or legacy 'image').
    Returns JSON with prediction or friendly error message.
    """
    file = request.files.get("file") or request.files.get("image")
    if not file:
        return jsonify({"status": "error", "message": "No image file provided. Send form-data with key 'file'."}), 400

    if file.filename == "":
        return jsonify({"status": "error", "message": "Uploaded file has no filename."}), 400

    try:
        raw = file.read()
        img = Image.open(io.BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")

        if classifier and getattr(classifier, "model", None):
            logger.info("Running real model prediction.")
            result = classifier.predict(img)
        else:
            logger.info("Running demo prediction (model not loaded).")
            demo_diseases = [
                "Tomato: Early Blight",
                "Potato: Late Blight",
                "Apple: Apple Scab",
                "Tomato: Healthy",
                "Potato: Healthy",
            ]
            conf = round(random.uniform(0.70, 0.95), 4)
            result = {
                "disease": random.choice(demo_diseases),
                "confidence": conf,
                "confidence_percentage": round(conf * 100, 2),
                "treatment": "Demo: model not available.",
                "severity": random.choice(["High", "Medium", "Low"]),
                "prediction_time": round(random.uniform(0.1, 0.5), 3),
                "model_version": "DEMO",
                "is_confident": True,
            }

        # Log prediction (best-effort)
        try:
            farmer_id = get_farmer_id()
            if db_manager:
                db_manager.add_farmer(farmer_id, name="Anonymous")
                # Try to save prediction; use class id if available
                predicted_id = result.get("class_id") or result.get("predicted_disease_id") or 1
                db_manager.save_prediction(
                    farmer_id=farmer_id,
                    predicted_disease_id=int(predicted_id),
                    confidence=float(result.get("confidence", 0)),
                    model_version=result.get("model_version", "unknown"),
                    prediction_time=float(result.get("prediction_time", 0)),
                    image_file=file.filename,
                )
                logger.info("Prediction logged for farmer %s", farmer_id)
        except Exception:
            logger.exception("Failed to log prediction to database.")

        return jsonify({"status": "success", **result}), 200

    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/chat", methods=["POST"])
def chat_with_ai_api():
    """
    Accepts JSON: { "message": "...", "crop": "Tomato", "language": "English", "disease_context": "..." }
    """
    data = request.get_json(silent=True) or {}
    farmer_query = data.get("message") or data.get("query")
    crop_type = data.get("crop", "General")
    disease_context = data.get("disease_context")
    language = data.get("language", "English")

    if not farmer_query:
        return jsonify({"status": "error", "message": "No message provided."}), 400

    if not chatbot:
        # fallback: return a helpful static response instead of failing
        fallback = (
            "Chat service currently unavailable. "
            "You can still use the prediction endpoint for disease detection."
        )
        return jsonify({"status": "error", "message": fallback}), 503

    try:
        response_text, response_time = chatbot.generate_response(
            farmer_query=farmer_query,
            crop_type=crop_type,
            disease_name=disease_context,
            language=language,
        )

        # log chatbot interaction (best-effort)
        try:
            farmer_id = get_farmer_id()
            if db_manager:
                db_manager.log_chatbot_interaction(
                    farmer_id=farmer_id,
                    query=farmer_query,
                    response=response_text,
                    language=language,
                    response_time=response_time,
                )
        except Exception:
            logger.exception("Failed to log chatbot interaction.")

        return jsonify({"status": "success", "response": response_text, "responseTime": round(response_time, 2)}), 200
    except Exception as e:
        logger.exception("Chat endpoint failed: %s", e)
        return jsonify({"status": "error", "message": f"Chat failed: {str(e)}"}), 500


@app.route("/api/analytics/summary", methods=["GET"])
def get_analytics_summary_api():
    if not analytics_engine:
        return jsonify({"status": "error", "message": "Analytics engine not initialized."}), 503
    try:
        summary = analytics_engine.get_dashboard_metrics()
        return jsonify({"status": "success", **summary}), 200
    except Exception:
        logger.exception("Failed to fetch analytics summary.")
        return jsonify({"status": "error", "message": "Could not fetch analytics."}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.exception("Internal server error: %s", error)
    return jsonify({"status": "error", "message": "Internal server error"}), 500


# ----- Main -----
if __name__ == "__main__":
    logger.info("Starting FarmAI API server...")
    load_global_resources()
    # quick health print
    logger.info("Model loaded: %s", bool(classifier and getattr(classifier, "model", None)))
    logger.info("Chatbot ready: %s", bool(chatbot))
    logger.info("Database ready: %s", bool(db_manager))
    app.run(host="0.0.0.0", port=5050, debug=False)
