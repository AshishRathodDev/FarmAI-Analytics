import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model('models/crop_disease_classifier_final.h5')
with open('models/class_indices.json', 'r') as f:
    class_names = list(json.load(f).keys())

def predict(image):
    """Predict disease from image"""
    img = image.resize((160, 160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]
    
    # Format results
    results = {}
    for i in range(len(class_names)):
        disease_name = class_names[i].replace('___', ': ').replace('_', ' ').title()
        results[disease_name] = float(predictions[i])
    
    return results

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=gr.Label(num_top_classes=5, label="Disease Predictions"),
    title="🌾 FarmAI - Crop Disease Detector",
    description="Upload a clear image of an affected leaf for AI-powered disease detection.\n\nModel: EfficientNetB0 | Accuracy: 60%+",
    examples=["examples/tomato_leaf.jpg"] if os.path.exists("examples") else None,
    theme=gr.themes.Soft()
)

demo.launch()
