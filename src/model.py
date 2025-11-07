"""
Model architecture for Syngenta Crop Disease Classification

Defines the CNN model using transfer learning with EfficientNetB0


"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from . import config

# Set random seed for reproducibility
tf.random.set_seed(config.RANDOM_SEED)

def build_transfer_learning_model(num_classes):
    """
    Builds a transfer learning model using EfficientNetB0 as the base.
    
    Args:
        num_classes (int): The number of output classes for the classifier.
        
    Returns:
        keras.Model: The compiled Keras model.
    """
    print("\n======================================================================")
    print("BUILDING MODEL: EfficientNetB0 (Transfer Learning)")
    print("======================================================================")
    
    # Load the pre-trained EfficientNetB0 model
    # include_top=False to remove the classification head
    # weights='imagenet' to use ImageNet pre-trained weights
    base_model = EfficientNetB0(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights='imagenet' if config.USE_PRETRAINED_WEIGHTS else None
    )
    
    # Freeze the base model's layers initially if configured
    base_model.trainable = not config.FREEZE_BASE_MODEL
    
    # Create the functional model
    inputs = keras.Input(shape=config.INPUT_SHAPE, name="input_layer")
    
    # EfficientNet expects inputs to be preprocessed (rescaled to [-1, 1]).
    # Our ImageDataGenerator will rescale to [0, 1].
    # The `EfficientNetB0` application itself has `preprocess_input`
    # but when used as a base, the scaling from [0,1] is generally acceptable.
    
    x = base_model(inputs, training=False) # Important: call base_model in inference mode if frozen
    
    # Add custom classification head
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="dropout_1")(x)
    x = layers.Dense(config.DENSE_UNITS, activation=config.ACTIVATION, name="dense_1")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation=config.FINAL_ACTIVATION, name="output_layer")(x)
    
    model = keras.Model(inputs, outputs, name=config.MODEL_ARCHITECTURE)
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.PHASE1_LEARNING_RATE),
        loss=config.LOSS_FUNCTION,
        metrics=[config.METRICS[0], keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    print(f"\n✓ Model Architecture: {config.MODEL_ARCHITECTURE}")
    print(f"  - Total Parameters: {model.count_params():,}")
    print(f"  - Trainable Parameters (initial): {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    print(f"  - Output Classes: {num_classes}")
    
    return model

def get_last_conv_layer_name(model):
    """
    Finds the name of the last suitable layer for Grad-CAM in the EfficientNetB0 base model.
    
    Args:
        model (keras.Model): The compiled Keras model.
        
    Returns:
        str: Name of the last convolutional/activation layer suitable for Grad-CAM, or None.
    """
    base_model = model.get_layer('efficientnetb0') # Get the EfficientNetB0 layer by name

    if base_model is None:
        print("Warning: EfficientNetB0 base model layer not found in the model for Grad-CAM.")
        return None
        
    # Common layers for Grad-CAM in EfficientNetB0:
    
    candidate_layers = ['block6a_expand_activation', 'block7a_project_bn', 'top_activation'] 
    
    for layer_name_candidate in candidate_layers:
        try:
            # Check if this layer exists within the base_model and has a 4D output
            layer = base_model.get_layer(layer_name_candidate)
            if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                print(f"✓ Found Grad-CAM layer: {layer_name_candidate}")
                return layer_name_candidate
        except ValueError:
            # Layer not found, try next candidate
            continue
            
    print("Warning: Could not find a suitable Grad-CAM layer within EfficientNetB0's base model. Grad-CAM may not function.")
    return None

if __name__ == "__main__":
    # Smoke test the model building
    print("Testing Model Building...")
    
    # Need a dummy class count
    dummy_num_classes = 12 
    model = build_transfer_learning_model(dummy_num_classes)
    
    print("\nModel Summary:")
    model.summary()
    
    last_conv_layer = get_last_conv_layer_name(model)
    print(f"\nDetected last convolutional layer for Grad-CAM: {last_conv_layer}")
    
    print("\n✓ Model building test completed successfully!")