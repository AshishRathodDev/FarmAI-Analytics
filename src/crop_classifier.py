"""
FarmAI Analytics Platform - Crop Disease Classifier
TensorFlow-based disease detection from crop images
"""

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json
import time
import logging
from pathlib import Path
from typing import Dict, Union, Tuple

# Setup logging
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CropDiseaseClassifier:
    """
    Load trained TensorFlow model and perform disease predictions
    """
    
    def __init__(self, model_path: str = 'models/crop_disease_model.h5'):
        """
        Initialize classifier with trained model
        
        Args:
            model_path: Path to trained .h5 model file
        """
        self.model_path = model_path
        self.model = None
        self.class_indices = {}
        self.image_size = (224, 224)
        
        # Default disease classes (PlantVillage dataset)
        self.disease_classes = {
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
        
        # Treatment database
        self.treatment_database = self._init_treatment_database()
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained TensorFlow model"""
        try:
            if Path(self.model_path).exists():
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"✅ Model loaded successfully from {self.model_path}")
                
                # Try to load class indices if available
                class_indices_path = self.model_path.replace('.h5', '_class_indices.json')
                if Path(class_indices_path).exists():
                    with open(class_indices_path, 'r') as f:
                        self.class_indices = json.load(f)
                        logger.info("✅ Class indices loaded")
            else:
                logger.warning(f"⚠️ Model file not found: {self.model_path}")
                logger.info("Running in DEMO mode - predictions will be simulated")
                self.model = None
        
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            self.model = None
    
    def preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image_input: Image file path, PIL Image, or numpy array
        
        Returns:
            Preprocessed image array
        """
        try:
            # Load image based on input type
            if isinstance(image_input, str):
                # File path
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Could not load image from {image_input}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = np.array(image_input.convert('RGB'))
            elif isinstance(image_input, np.ndarray):
                # Already numpy array
                image = image_input
            else:
                raise ValueError(f"Unsupported image type: {type(image_input)}")
            
            # Resize to model input size
            image = cv2.resize(image, self.image_size)
            
            # Normalize to [0, 1]
            image = image.astype('float32') / 255.0
            
            return image
        
        except Exception as e:
            logger.error(f"❌ Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Predict disease from image
        
        Args:
            image_input: Image to analyze
        
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            image_array = self.preprocess_image(image_input)
            
            if self.model is not None:
                # Real prediction with model
                image_batch = np.expand_dims(image_array, axis=0)
                predictions = self.model.predict(image_batch, verbose=0)
                
                # Get top prediction
                predicted_class_idx = int(np.argmax(predictions[0]))
                confidence = float(np.max(predictions[0]))
                
                # Get top 3 predictions
                top3_indices = np.argsort(predictions[0])[-3:][::-1]
                top3_predictions = [
                    {
                        'class_id': int(idx),
                        'disease': self._format_disease_name(self.disease_classes.get(idx, 'Unknown')),
                        'confidence': float(predictions[0][idx]),
                        'confidence_percentage': float(predictions[0][idx]) * 100
                    }
                    for idx in top3_indices
                ]
            else:
                # Demo mode - simulated predictions
                logger.info("⚠️ Using demo mode prediction")
                predicted_class_idx = np.random.randint(0, len(self.disease_classes))
                confidence = float(np.random.uniform(0.65, 0.98))
                
                top3_predictions = [
                    {
                        'class_id': predicted_class_idx,
                        'disease': self._format_disease_name(self.disease_classes[predicted_class_idx]),
                        'confidence': confidence,
                        'confidence_percentage': confidence * 100
                    }
                ]
            
            prediction_time = time.time() - start_time
            
            # Get disease name
            disease_name = self.disease_classes.get(predicted_class_idx, 'Unknown Disease')
            formatted_disease_name = self._format_disease_name(disease_name)
            
            # Get treatment recommendation
            treatment = self._get_treatment_recommendation(formatted_disease_name)
            
            # Determine severity based on confidence and disease type
            severity = self._determine_severity(confidence, formatted_disease_name)
            
            # Build result dictionary
            result = {
                'disease': formatted_disease_name,
                'disease_raw': disease_name,
                'class_id': predicted_class_idx,
                'confidence': round(confidence, 4),
                'confidence_percentage': round(confidence * 100, 2),
                'severity': severity,
                'treatment': treatment,
                'top3_predictions': top3_predictions,
                'prediction_time': round(prediction_time, 3),
                'model_version': '1.0',
                'is_confident': confidence >= 0.65,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"✅ Prediction: {formatted_disease_name} ({result['confidence_percentage']}%)")
            return result
        
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            return {
                'error': str(e),
                'disease': 'Error',
                'confidence': 0,
                'confidence_percentage': 0,
                'treatment': 'Unable to analyze image. Please try again with a clearer photo.',
                'prediction_time': time.time() - start_time
            }
    
    def predict_batch(self, images: list) -> list:
        """
        Predict diseases for multiple images
        
        Args:
            images: List of images to analyze
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for img in images:
            result = self.predict(img)
            results.append(result)
        return results
    
    def _format_disease_name(self, disease_name: str) -> str:
        """Format disease name for display"""
        # Replace underscores and format
        formatted = disease_name.replace('___', ': ').replace('_', ' ')
        
        # Capitalize properly
        words = formatted.split()
        formatted = ' '.join(word.capitalize() for word in words)
        
        return formatted
    
    def _determine_severity(self, confidence: float, disease_name: str) -> str:
        """
        Determine disease severity level
        
        Args:
            confidence: Prediction confidence
            disease_name: Name of detected disease
        
        Returns:
            Severity level: High, Medium, or Low
        """
        # Check if healthy
        if 'healthy' in disease_name.lower():
            return 'Low'
        
        # High severity diseases
        high_severity_keywords = ['blight', 'rot', 'wilt', 'virus', 'mosaic']
        if any(keyword in disease_name.lower() for keyword in high_severity_keywords):
            if confidence >= 0.80:
                return 'High'
            else:
                return 'Medium'
        
        # Medium severity diseases
        medium_severity_keywords = ['spot', 'mold', 'rust', 'scab']
        if any(keyword in disease_name.lower() for keyword in medium_severity_keywords):
            return 'Medium'
        
        # Default based on confidence
        if confidence >= 0.85:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_treatment_recommendation(self, disease_name: str) -> str:
        """Get treatment recommendation for disease"""
        
        # Check treatment database
        for key, treatment in self.treatment_database.items():
            if key.lower() in disease_name.lower():
                return treatment
        
        # Default recommendations based on disease type
        if 'blight' in disease_name.lower():
            return """
**Blight Treatment Plan:**
1. Remove and destroy infected plant parts immediately
2. Apply copper-based fungicide (e.g., Copper Oxychloride)
3. Spray every 7-10 days until symptoms disappear
4. Improve air circulation around plants
5. Avoid overhead watering
6. Apply preventive fungicides before rainy season

**Cost:** ₹300-500 per acre | **Recovery:** 2-3 weeks
"""
        
        elif 'spot' in disease_name.lower():
            return """
**Bacterial/Fungal Spot Treatment:**
1. Remove infected leaves
2. Apply Mancozeb or Chlorothalonil fungicide
3. Spray every 5-7 days
4. Ensure good drainage
5. Practice crop rotation next season

**Cost:** ₹200-400 per acre | **Recovery:** 1-2 weeks
"""
        
        elif 'healthy' in disease_name.lower():
            return """
**Plant is Healthy! 🌱**

Preventive Care:
- Continue regular monitoring
- Maintain proper watering schedule
- Apply balanced fertilizers
- Practice good field sanitation
- Monitor for early disease signs
"""
        
        else:
            return """
**General Disease Management:**
1. Isolate affected plants
2. Remove infected parts
3. Apply appropriate fungicide/pesticide
4. Improve field hygiene
5. Consult local agricultural expert for specific treatment

**Recommendation:** Visit nearest agricultural extension center for precise diagnosis
"""
    
    def _init_treatment_database(self) -> Dict[str, str]:
        """Initialize treatment recommendations database"""
        return {
            'Early Blight': """
**Early Blight Treatment:**
1. Remove lower infected leaves
2. Apply Mancozeb 75% WP @ 2g/liter
3. Spray Chlorothalonil every 7-10 days
4. Maintain plant spacing for air circulation
5. Mulch to prevent soil splash

**Products:** Indofil M-45, Kavach, Blitox
**Cost:** ₹400-600/acre | **Recovery:** 2-3 weeks
""",
            
            'Late Blight': """
**Late Blight Treatment (URGENT):**
1. Act immediately - highly destructive disease
2. Apply Metalaxyl + Mancozeb (Ridomil Gold)
3. Spray every 5-7 days during humid weather
4. Remove and burn infected plants
5. Improve drainage

**Products:** Ridomil Gold, Curzate, Secure
**Cost:** ₹600-800/acre | **Recovery:** 2-4 weeks
""",
            
            'Bacterial Spot': """
**Bacterial Spot Treatment:**
1. Remove infected plant parts
2. Apply copper-based bactericide
3. Spray Streptocycline (antibiotic) if severe
4. Avoid working with wet plants
5. Disinfect tools with alcohol

**Products:** Kocide, Agrimycin, Plantomycin
**Cost:** ₹300-500/acre | **Recovery:** 2-3 weeks
""",
            
            'Leaf Mold': """
**Leaf Mold Treatment:**
1. Improve greenhouse ventilation
2. Reduce humidity below 85%
3. Apply Difenoconazole or Azoxystrobin
4. Remove infected leaves
5. Space plants properly

**Products:** Score, Amistar, Cabrio
**Cost:** ₹400-600/acre | **Recovery:** 1-2 weeks
""",
            
            'Septoria Leaf Spot': """
**Septoria Leaf Spot Treatment:**
1. Remove and destroy infected leaves
2. Apply Chlorothalonil fungicide
3. Spray every 7-14 days
4. Rotate crops annually
5. Use resistant varieties

**Products:** Bravo, Daconil, Kavach
**Cost:** ₹350-550/acre | **Recovery:** 2-3 weeks
""",
            
            'Spider Mites': """
**Spider Mite Control:**
1. Spray with water to dislodge mites
2. Apply Abamectin or Propargite
3. Use neem oil for organic control
4. Introduce predatory mites if available
5. Maintain adequate moisture

**Products:** Vertimec, Omite, Neem oil
**Cost:** ₹250-400/acre | **Recovery:** 1-2 weeks
"""
        }
    
    def get_model_info(self) -> Dict:
        """Get model information and statistics"""
        if self.model:
            return {
                'model_loaded': True,
                'model_path': self.model_path,
                'input_shape': self.model.input_shape,
                'output_classes': len(self.disease_classes),
                'total_parameters': self.model.count_params() if hasattr(self.model, 'count_params') else 'N/A'
            }
        else:
            return {
                'model_loaded': False,
                'status': 'Demo mode - model file not found',
                'model_path': self.model_path
            }


# Testing code
if __name__ == "__main__":
    classifier = CropDiseaseClassifier()
    
    print("Model Info:")
    print(classifier.get_model_info())
    
    print("\nTreatment Database:")
    for disease, treatment in list(classifier.treatment_database.items())[:2]:
        print(f"\n{disease}:")
        print(treatment[:100] + "...")