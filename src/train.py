"""
Training pipeline for Syngenta Crop Disease Classification

Handles model training including two-phase approach and callbacks

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import json
import numpy as np

from . import config 
from .model import build_transfer_learning_model


tf.random.set_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

def plot_training_history(history, save_path=None):
    """
    Plots training and validation accuracy and loss over epochs.
    
    Args:
        history (keras.callbacks.History): Training history object.
        save_path (Path, optional): Path to save the plot. Defaults to None.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✓ Training history plot saved to: {save_path}")
    plt.show()

def save_training_history(history, save_path=None):
    """
    Saves the training history to a JSON file.
    
    Args:
        history (keras.callbacks.History): Training history object.
        save_path (Path, optional): Path to save the JSON file. Defaults to None.
    """
    save_path = save_path or config.TRAINING_HISTORY_PATH
    with open(save_path, 'w') as f:
        # Convert numpy floats to Python floats for JSON serialization
        serializable_history = {k: [float(v) for v in val] for k, val in history.history.items()}
        json.dump(serializable_history, f, indent=4)
    print(f"✓ Training history saved to: {save_path}")

def train_model_pipeline(train_generator, val_generator, num_classes):
    """
    Orchestrates the two-phase training process for the crop disease classifier.
    
    Args:
        train_generator (ImageDataGenerator): Training data generator.
        val_generator (ImageDataGenerator): Validation data generator.
        num_classes (int): Number of output classes.
        
    Returns:
        keras.Model: The trained model (best weights loaded).
        keras.callbacks.History: Combined training history.
    """
    print("\n======================================================================")
    print("STARTING MODEL TRAINING PIPELINE")
    print("======================================================================")

    # Build the model initially
    model = build_transfer_learning_model(num_classes)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor=config.MONITOR_METRIC,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=config.EARLY_STOPPING_RESTORE_BEST,
            verbose=config.VERBOSE
        ),
        ReduceLROnPlateau(
            monitor=config.MONITOR_METRIC,
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.REDUCE_LR_MIN_LR,
            verbose=config.VERBOSE
        ),
        ModelCheckpoint(
            config.CHECKPOINT_FILEPATH,
            monitor=config.MONITOR_METRIC,
            save_best_only=config.SAVE_BEST_ONLY,
            mode=config.CHECKPOINT_MODE,
            verbose=config.VERBOSE
        )
    ]

    # --- Phase 1: Train top layers with frozen base ---
    print(f"\n[Phase 1] Training top layers (base frozen) for {config.PHASE1_EPOCHS} epochs...")
    print(f"  Initial learning rate: {config.PHASE1_LEARNING_RATE}")
    
    # Freeze the base model explicitly
    for layer in model.layers: 
        if layer.name.startswith("efficientnetb0"): 
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.PHASE1_LEARNING_RATE),
        loss=config.LOSS_FUNCTION,
        metrics=[config.METRICS[0], keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    # Check if GPU is available
    if not tf.config.list_physical_devices('GPU'):
        print("\nWARNING: GPU not detected. Training on CPU will be significantly slower.")
        print(f"  Consider reducing epochs for smoke test (current: {config.PHASE1_EPOCHS}+{config.PHASE2_EPOCHS}) or using a GPU.")
        # For a production script, we might exit or force reduced epochs here if not interactive.

    history_phase1 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.PHASE1_EPOCHS,
        callbacks=callbacks,
        verbose=config.VERBOSE
    )
    
    # Load the best weights saved during phase 1 before proceeding to phase 2
    model = tf.keras.models.load_model(config.CHECKPOINT_FILEPATH)
    print(f"\n✓ Best model from Phase 1 loaded from {config.CHECKPOINT_FILEPATH}")

    # --- Phase 2: Fine-tuning with unfrozen layers ---
    if config.PHASE2_EPOCHS > 0 and config.UNFREEZE_LAYERS > 0:
        print(f"\n[Phase 2] Fine-tuning (unfreezing last {config.UNFREEZE_LAYERS} layers) for {config.PHASE2_EPOCHS} epochs...")
        print(f"  Fine-tuning learning rate: {config.PHASE2_LEARNING_RATE}")

        # Unfreeze the base model
        base_model = model.layers[1] # This is EfficientNetB0
        base_model.trainable = True

        # Freeze all layers except the last 'UNFREEZE_LAYERS' in the base model
        for layer in base_model.layers[:-config.UNFREEZE_LAYERS]:
            layer.trainable = False

        # Recompile with a lower learning rate for fine-tuning
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.PHASE2_LEARNING_RATE),
            loss=config.LOSS_FUNCTION,
            metrics=[config.METRICS[0], keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        # Training continues from the state after phase 1
        history_phase2 = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=config.PHASE2_EPOCHS,
            callbacks=callbacks,
            verbose=config.VERBOSE 
        )
        
        # Combine histories
        combined_history = {key: history_phase1.history.get(key, []) + history_phase2.history.get(key, [])
                            for key in set(history_phase1.history) | set(history_phase2.history)}
        
        # Create a dummy History object for combined history
        class CombinedHistory(keras.callbacks.History):
            def __init__(self, history_dict):
                super().__init__()
                self.history = history_dict
        
        combined_history_obj = CombinedHistory(combined_history)

    else:
        print("\nSkipping Phase 2 (fine-tuning) as per configuration.")
        combined_history_obj = history_phase1

    # Load the best model saved during combined training phases
    model = tf.keras.models.load_model(config.CHECKPOINT_FILEPATH)
    print(f"\n✓ Final best model loaded from {config.CHECKPOINT_FILEPATH}")
    
    # Save the final trained model
    model.save(config.FINAL_MODEL_PATH)
    print(f"✓ Final trained model saved to: {config.FINAL_MODEL_PATH}")

    return model, combined_history_obj

if __name__ == "__main__":
    # Smoke test training (requires data_utils to run first)
    print("Testing Training Pipeline (Smoke Test)...")
    
    # Use data_utils to get generators
    from src.data_utils import DataPipeline
    data_pipeline = DataPipeline()
    
    # Ensure dataset structure is verified and splits are created
    data_pipeline.verify_dataset_structure()
    train_gen, val_gen, _, _ = data_pipeline.create_data_generators()
    
    # Temporarily override epochs for a quick smoke test
    original_phase1_epochs = config.PHASE1_EPOCHS
    original_phase2_epochs = config.PHASE2_EPOCHS
    config.PHASE1_EPOCHS = 1
    config.PHASE2_EPOCHS = 1
    
    model, history = train_model_pipeline(train_gen, val_gen, len(train_gen.class_indices))
    plot_training_history(history)
    
    # Restore original epochs
    config.PHASE1_EPOCHS = original_phase1_epochs
    config.PHASE2_EPOCHS = original_phase2_epochs
    
    print("\n✓ Training pipeline smoke test completed successfully!")
    
    