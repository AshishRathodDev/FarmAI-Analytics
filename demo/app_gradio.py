"""
Gradio Demo Application for Syngenta Crop Disease Classification

Provides an interactive web interface to predict disease from uploaded images.

"""

import gradio as gr
import tensorflow as tf
import numpy as np
from pathlib import Path
import json

import sys
# Added  src to path to import config and data_utils
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import config
from src.data_utils import preprocess_image # Re-use preprocessing logic

# Global variables for model and class mapping
model = None
class_indices = None
idx_to_class_name = None

def load_artifacts():
    """Load the trained model and class indices."""
    global model, class_indices, idx_to_class_name
    
    print("\nLoading model and class indices for Gradio app...")
    
    # Load model
    if not config.FINAL_MODEL_PATH.exists():
        print(f"Error: Model not found at {config.FINAL_MODEL_PATH}. Please train the model first.")
        sys.exit(1)
    model = tf.keras.models.load_model(config.FINAL_MODEL_PATH)
    print(f"✓ Model loaded from: {config.FINAL_MODEL_PATH}")
    
    # Load class indices
    if not config.CLASS_INDICES_PATH.exists():
        print(f"Error: Class indices not found at {config.CLASS_INDICES_PATH}. Please train the model first.")
        sys.exit(1)
    with open(config.CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    idx_to_class_name = {v: k for k, v in class_indices.items()}
    print(f"✓ Class indices loaded. Total classes: {len(class_indices)}")
    
    print("✓ Gradio app artifacts loaded successfully.")

def predict_image(image):
    """
    Predicts the crop disease from an input image.
    
    Args:
        image (np.array): Input image as a NumPy array (from Gradio).
        
    Returns:
        dict: A dictionary of class probabilities.
    """
    if image is None:
        return {}
    
    if model is None or class_indices is None:
        load_artifacts() # Lazy load if not already loaded

    # Gradio often returns a NumPy array for Image component.
    # Convert numpy image (RGB) to PIL Image, then preprocess
    from PIL import Image as PILImage
    pil_image = PILImage.fromarray(image.astype('uint8'), 'RGB')
    
    # Preprocess image using the same function as during training
    img_tensor = tf.convert_to_tensor(pil_image, dtype=tf.float32)
    img_tensor = tf.image.resize(img_tensor, config.IMG_SIZE)
    img_tensor = img_tensor / 255.0  # Normalize to [0, 1]
    img_tensor = tf.expand_dims(img_tensor, axis=0) # Add batch dimension

    # Make prediction
    predictions = model.predict(img_tensor)[0] # Get probabilities for the single image
    
    # Format results
    predicted_probs = {idx_to_class_name[i].replace('___', ' '): float(predictions[i]) for i in range(len(class_indices))}
    
    return predicted_probs

def get_demo_examples():
    """
    Collects example image paths for the Gradio demo.
    
    Returns:
        list: List of lists, where each inner list contains an image path.
    """
    examples = []
    
    # Ensure data directory exists
    if not config.RAW_DATA_DIR.exists():
        print(f"Warning: Demo examples directory not found at {config.RAW_DATA_DIR}.")
        print("Skipping example image loading for Gradio.")
        return []
    
    # Get a few random images from different classes
    class_dirs = [d for d in config.RAW_DATA_DIR.iterdir() if d.is_dir()]
    selected_classes = class_dirs[:min(5, len(class_dirs))] # Take first 5 classes
    
    for class_path in selected_classes:
        images = list(class_path.glob('*.[jJ][pP][gG]')) + list(class_path.glob('*.[pP][nN][gG]'))
        if images:
            examples.append([str(images[0])]) # Take one image per class
            
    print(f"✓ Loaded {len(examples)} example images for Gradio demo.")
    return examples

def main():
    """Main function to launch the Gradio app."""
    load_artifacts() # Load artifacts once at startup

    # Define Gradio interface
    iface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="numpy", label="Upload Leaf Image"),
        outputs=gr.Label(num_top_classes=3, label="Top 3 Predictions"),
        title="Syngenta Crop Disease Classifier",
        description="Upload an image of a plant leaf to classify potential diseases. "
                    "The model predicts the top 3 most likely diseases with confidence scores.",
        examples=get_demo_examples(),
        live=False,
        allow_flagging='never'
    )
    
    print("\n================================================================================")
    print(" " * 25 + "LAUNCHING GRADIO DEMO APP")
    print("================================================================================")
    print(f"App will run on http://127.0.0.1:{config.DEMO_PORT}")
    if config.DEMO_SHARE:
        print("Sharing link will be generated (may take a moment)...")
    
    # Launch the app
    iface.launch(server_port=config.DEMO_PORT, share=config.DEMO_SHARE)

if __name__ == "__main__":
    main()