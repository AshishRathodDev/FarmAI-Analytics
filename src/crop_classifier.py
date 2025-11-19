"""
FarmAI Analytics Platform - Crop Disease Classifier
TensorFlow-based disease detection from crop images
"""

from pathlib import Path
from typing import Dict, Union, Tuple, Any
import json
import time
import logging

# Optional imports that may fail on some systems (TensorFlow on mac M1 etc.)
try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import numpy as np
except Exception:
    raise

# cv2 is optional for some environments; fallback to PIL processing if not available
try:
    import cv2
except Exception:
    cv2 = None

from PIL import Image

# Ensure logs dir exists
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "app.log"

# Setup logging (safe even if main app configures logging earlier)
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CropDiseaseClassifier:
    """
    Load trained TensorFlow model and perform disease predictions.

    If TensorFlow or the model file is not available, the classifier will run in DEMO mode
    and return simulated predictions (useful for UI testing).
    """

    def __init__(self, model_path: Union[str, Path] = "models/crop_disease_model.h5",
                 image_size: Tuple[int, int] = (224, 224),
                 confidence_threshold: float = 0.65):
        self.model_path = Path(model_path)
        self.model = None
        self.class_indices: Dict[str, Any] = {}
        self.image_size = tuple(image_size)
        self.confidence_threshold = float(confidence_threshold)

        # Default disease classes (fallback if class indices are not provided)
        self.disease_classes: Dict[int, str] = {
            0: 'Apple___Apple_scab',
            1: 'Apple___Black_rot',
            2: 'Apple___Cedar_apple_rust',
            3: 'Apple___healthy',
            4: 'Grape___Black_rot',
            5: 'Grape___Esca_(Black_Measles)',
            6: 'Grape___Leaf_blight',
            7: 'Grape___healthy',
            8: 'Orange___Haunglongbing',
            9: 'Peach___Bacterial_spot',
            10: 'Peach___healthy',
            11: 'Pepper,_bell___Bacterial_spot',
            12: 'Pepper,_bell___healthy',
            13: 'Potato___Early_blight',
            14: 'Potato___Late_blight',
            15: 'Potato___healthy',
            16: 'Tomato___Bacterial_spot',
            17: 'Tomato___Early_blight',
            18: 'Tomato___Late_blight',
            19: 'Tomato___Leaf_Mold',
            20: 'Tomato___Septoria_leaf_spot',
            21: 'Tomato___Spider_mites',
            22: 'Tomato___Target_Spot',
            23: 'Tomato___Yellow_Leaf_Curl_Virus',
            24: 'Tomato___Tomato_mosaic_virus',
            25: 'Tomato___healthy'
        }

        # Treatment templates are provided by the app; classifier calls a helper to fetch them
        self.treatment_database = self._init_treatment_database()

        # Load model and class indices (if present)
        self._load_model_and_classes()

    def _load_model_and_classes(self) -> None:
        """
        Load TF model if available and load class indices JSON if present.
        Handles multiple formats for class indices (index->name or name->index).
        """
        # Load class indices from sibling JSON file (modelname_class_indices.json) or generic class_indices.json
        candidates = [
            self.model_path.with_name(self.model_path.stem + "_class_indices.json"),
            self.model_path.with_suffix("_class_indices.json"),
            self.model_path.parent / "class_indices.json"
        ]
        for p in candidates:
            try:
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        j = json.load(f)
                        # detect format: if keys are numeric strings -> invert if necessary
                        if all(k.isdigit() for k in map(str, j.keys())):
                            # already index -> name mapping
                            # convert keys to int
                            self.class_indices = {int(k): v for k, v in j.items()}
                        else:
                            # assume name -> index; invert to index -> name
                            inv = {}
                            for name, idx in j.items():
                                try:
                                    inv[int(idx)] = name
                                except Exception:
                                    # if idx not an int, ignore
                                    continue
                            if inv:
                                self.class_indices = inv
                            else:
                                # fallback: keep original
                                self.class_indices = j
                    logger.info("Loaded class indices from %s", str(p))
                    break
            except Exception as e:
                logger.warning("Failed to load class indices from %s: %s", str(p), str(e))
                continue

        # If class_indices loaded and keys are ints, update disease_classes fallback
        if isinstance(self.class_indices, dict) and all(isinstance(k, int) for k in self.class_indices.keys()):
            # update disease_classes mapping for better display
            for idx, name in self.class_indices.items():
                try:
                    self.disease_classes[int(idx)] = name
                except Exception:
                    continue

        # Load TensorFlow model if possible
        if tf is None:
            logger.warning("TensorFlow is not available in this environment. Running in demo mode.")
            self.model = None
            return

        try:
            if self.model_path.exists():
                # Use absolute path string for TF
                self.model = tf.keras.models.load_model(str(self.model_path))
                logger.info("Model loaded successfully from %s", str(self.model_path))
            else:
                logger.warning("Model file not found at %s. Running in demo mode.", str(self.model_path))
                self.model = None
        except Exception as e:
            logger.error("Error loading model from %s : %s", str(self.model_path), str(e))
            self.model = None

    def preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model input.
        Accepts file path string, PIL Image, or numpy array.
        Returns numpy array shaped (H, W, C) normalized to [0,1].
        """
        try:
            # Load image according to type
            if isinstance(image_input, str):
                # path
                if cv2 is not None:
                    image_bgr = cv2.imread(image_input)
                    if image_bgr is None:
                        raise ValueError(f"Could not read image at {image_input}")
                    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                else:
                    image = Image.open(image_input).convert("RGB")
                    image = np.array(image)
            elif isinstance(image_input, Image.Image):
                image = np.array(image_input.convert("RGB"))
            elif isinstance(image_input, np.ndarray):
                image = image_input
                # if grayscale expand dims
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
            else:
                raise ValueError(f"Unsupported image type: {type(image_input)}")

            # Resize with cv2 if available for speed, else use PIL
            target_w, target_h = self.image_size
            if cv2 is not None and isinstance(image, np.ndarray):
                image_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                pil = Image.fromarray(image)
                pil = pil.resize((target_w, target_h), Image.BILINEAR)
                image_resized = np.array(pil)

            image_resized = image_resized.astype("float32") / 255.0

            # Ensure shape (H, W, 3)
            if image_resized.ndim == 2:
                image_resized = np.stack([image_resized] * 3, axis=-1)
            if image_resized.shape[-1] == 4:
                # drop alpha if present
                image_resized = image_resized[..., :3]

            return image_resized

        except Exception as e:
            logger.error("Error in preprocess_image: %s", str(e))
            raise

    def predict(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Predict disease from a single image. Returns a dictionary with standard fields.
        If model not loaded, returns a demo/simulated response.
        """
        start_time = time.time()
        try:
            img = self.preprocess_image(image_input)
            if self.model is not None:
                batch = np.expand_dims(img, axis=0)
                preds = self.model.predict(batch, verbose=0)[0]
                predicted_idx = int(np.argmax(preds))
                confidence = float(preds[predicted_idx])

                # top 3
                top3 = np.argsort(preds)[-3:][::-1]
                top3_predictions = []
                for idx in top3:
                    cid = int(idx)
                    name = self.disease_classes.get(cid, self.class_indices.get(cid, f"Class_{cid}"))
                    top3_predictions.append({
                        "class_id": cid,
                        "disease": self._format_disease_name(str(name)),
                        "confidence": float(preds[cid]),
                        "confidence_percentage": float(preds[cid]) * 100.0
                    })
            else:
                # Demo mode: random selection from disease_classes
                logger.info("Model not loaded, returning demo prediction")
                predicted_idx = int(np.random.randint(0, max(self.disease_classes.keys()) + 1))
                confidence = float(np.random.uniform(0.70, 0.95))
                top3_predictions = [{
                    "class_id": predicted_idx,
                    "disease": self._format_disease_name(self.disease_classes.get(predicted_idx, f"Class_{predicted_idx}")),
                    "confidence": confidence,
                    "confidence_percentage": confidence * 100.0
                }]

            prediction_time = time.time() - start_time
            disease_raw = self.disease_classes.get(predicted_idx, self.class_indices.get(predicted_idx, f"Class_{predicted_idx}"))
            disease_display = self._format_disease_name(str(disease_raw))
            treatment = self._get_treatment_recommendation(disease_display)
            severity = self._determine_severity(confidence, disease_display)

            result = {
                "disease": disease_display,
                "disease_raw": disease_raw,
                "class_id": int(predicted_idx),
                "confidence": round(confidence, 4),
                "confidence_percentage": round(confidence * 100.0, 2),
                "severity": severity,
                "treatment": treatment,
                "top3_predictions": top3_predictions,
                "prediction_time": round(prediction_time, 3),
                "model_version": "real" if self.model is not None else "DEMO",
                "is_confident": confidence >= self.confidence_threshold,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            logger.info("Prediction result: %s (%.2f%%)", result["disease"], result["confidence_percentage"])
            return result

        except Exception as e:
            logger.exception("Prediction error: %s", str(e))
            return {
                "error": str(e),
                "disease": "Error",
                "confidence": 0.0,
                "confidence_percentage": 0.0,
                "treatment": "Unable to analyze image. Please try again with a clearer photo.",
                "prediction_time": round(time.time() - start_time, 3)
            }

    def predict_batch(self, images: list) -> list:
        results = []
        for img in images:
            results.append(self.predict(img))
        return results

    def _format_disease_name(self, disease_name: str) -> str:
        formatted = disease_name.replace("___", ": ").replace("_", " ")
        formatted = " ".join(word.capitalize() for word in formatted.split())
        return formatted

    def _determine_severity(self, confidence: float, disease_name: str) -> str:
        if "healthy" in disease_name.lower():
            return "Low"
        high_severity_keywords = ["blight", "rot", "wilt", "virus", "mosaic"]
        if any(k in disease_name.lower() for k in high_severity_keywords):
            return "High" if confidence >= 0.80 else "Medium"
        medium_keywords = ["spot", "mold", "rust", "scab"]
        if any(k in disease_name.lower() for k in medium_keywords):
            return "Medium"
        return "Medium" if confidence >= 0.85 else "Low"

    def _get_treatment_recommendation(self, disease_name: str) -> str:
        for key, treatment in self.treatment_database.items():
            if key.lower() in disease_name.lower():
                return treatment
        # default generic advice
        return """
General Disease Management:
1. Remove infected plant parts and destroy them.
2. Improve air circulation and avoid overhead watering.
3. Consider applying appropriate fungicide or bactericide according to label instructions.
4. Consult local agricultural extension officers for region-specific products.
"""

    def _init_treatment_database(self) -> Dict[str, str]:
        return {
            "Early Blight": """
Early Blight Treatment:
1. Remove lower infected leaves.
2. Apply Mancozeb 75% WP @ recommended rate.
3. Spray at 7-10 day intervals until symptoms subside.
""",
            "Late Blight": """
Late Blight Treatment:
1. Act urgently; remove and burn severely infected plants.
2. Apply recommended systemic fungicides (e.g. Metalaxyl combinations).
"""
        }

    def get_model_info(self) -> Dict:
        if self.model is not None:
            try:
                input_shape = getattr(self.model, "input_shape", "N/A")
                params = self.model.count_params() if hasattr(self.model, "count_params") else "N/A"
            except Exception:
                input_shape = "N/A"
                params = "N/A"
            return {
                "model_loaded": True,
                "model_path": str(self.model_path),
                "input_shape": input_shape,
                "output_classes": len(self.disease_classes),
                "total_parameters": params
            }
        else:
            return {
                "model_loaded": False,
                "status": "Demo mode - model file not found or TensorFlow unavailable",
                "model_path": str(self.model_path)
            }


if __name__ == "__main__":
    # quick smoke tests (will run in demo mode if model/tf not available)
    classifier = CropDiseaseClassifier()
    print("Model info:", classifier.get_model_info())

    # sample test: create a small blank image and run predict
    img = Image.new("RGB", classifier.image_size, (128, 128, 128))
    result = classifier.predict(img)
    print("Sample prediction:", result)
