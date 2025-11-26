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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from .model import build_transfer_learning_model

tf.random.set_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

def plot_training_history(history, save_path=None):
    """
    Plots training and validation accuracy and loss over epochs.
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
        plt.savefig(str(save_path))  # Path to str
        print(f"✓ Training history plot saved to: {save_path}")
    plt.show()

def save_training_history(history, save_path=None):
    """
    Saves the training history to a JSON file.
    """
    save_path = save_path or config.TRAINING_HISTORY_PATH
    with open(str(save_path), 'w') as f:  # Path to str
        serializable_history = {k: [float(v) for v in val] for k, val in history.history.items()}
        json.dump(serializable_history, f, indent=4)
    print(f"✓ Training history saved to: {save_path}")

def train_model_pipeline(train_generator, val_generator, num_classes):
    """
    Orchestrates the two-phase training process for the crop disease classifier.
    """
    print("\n======================================================================")
    print("STARTING MODEL TRAINING PIPELINE")
    print("======================================================================")

    # Build model
    model = build_transfer_learning_model(num_classes)

    # Define callbacks (convert Paths to strings)
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
            str(config.CHECKPOINT_FILEPATH),   # Path to str
            monitor=config.MONITOR_METRIC,
            save_best_only=config.SAVE_BEST_ONLY,
            mode=config.CHECKPOINT_MODE,
            verbose=config.VERBOSE
        )
    ]

    # --- Phase 1: Train top layers with frozen base ---
    print(f"\n[Phase 1] Training top layers (base frozen) for {config.PHASE1_EPOCHS} epochs...")
    print(f"  Initial learning rate: {config.PHASE1_LEARNING_RATE}")

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

    if not tf.config.list_physical_devices('GPU'):
        print("\nWARNING: GPU not detected. Training on CPU will be significantly slower.")
        print(f"  Consider reducing epochs for smoke test (current: {config.PHASE1_EPOCHS}+{config.PHASE2_EPOCHS}) or using a GPU.")

    history_phase1 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.PHASE1_EPOCHS,
        callbacks=callbacks,
        verbose=config.VERBOSE
    )

    # Load best weights from Phase 1
    model = tf.keras.models.load_model(str(config.CHECKPOINT_FILEPATH))
    print(f"\n✓ Best model from Phase 1 loaded from {config.CHECKPOINT_FILEPATH}")

    # --- Phase 2: Fine-tuning ---
    if config.PHASE2_EPOCHS > 0 and config.UNFREEZE_LAYERS > 0:
        print(f"\n[Phase 2] Fine-tuning (unfreezing last {config.UNFREEZE_LAYERS} layers) for {config.PHASE2_EPOCHS} epochs...")
        print(f"  Fine-tuning learning rate: {config.PHASE2_LEARNING_RATE}")

        base_model = model.layers[1]
        base_model.trainable = True

        for layer in base_model.layers[:-config.UNFREEZE_LAYERS]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.PHASE2_LEARNING_RATE),
            loss=config.LOSS_FUNCTION,
            metrics=[config.METRICS[0], keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )

        history_phase2 = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=config.PHASE2_EPOCHS,
            callbacks=callbacks,
            verbose=config.VERBOSE
        )

        combined_history = {key: history_phase1.history.get(key, []) + history_phase2.history.get(key, [])
                            for key in set(history_phase1.history) | set(history_phase2.history)}

        class CombinedHistory(keras.callbacks.History):
            def __init__(self, history_dict):
                super().__init__()
                self.history = history_dict

        combined_history_obj = CombinedHistory(combined_history)

    else:
        print("\nSkipping Phase 2 (fine-tuning) as per configuration.")
        combined_history_obj = history_phase1

    # Load the best model saved during combined training phases
    model = tf.keras.models.load_model(str(config.CHECKPOINT_FILEPATH))
    print(f"\n✓ Final best model loaded from {config.CHECKPOINT_FILEPATH}")

    # Save the final trained model (Path to string)
    final_keras_path = str(config.FINAL_MODEL_PATH).replace('.h5', '.keras')
    model.save(final_keras_path)
    print(f"✓ Final trained model saved to: {final_keras_path}")



    return model, combined_history_obj

if __name__ == "__main__":
    print("Testing Training Pipeline (Smoke Test)...")
    from src.data_utils import DataPipeline
    data_pipeline = DataPipeline()
    data_pipeline.verify_dataset_structure()
    train_gen, val_gen, _, _ = data_pipeline.create_data_generators()
    original_phase1_epochs = config.PHASE1_EPOCHS
    original_phase2_epochs = config.PHASE2_EPOCHS
    config.PHASE1_EPOCHS = 1
    config.PHASE2_EPOCHS = 1
    model, history = train_model_pipeline(train_gen, val_gen, len(train_gen.class_indices))
    plot_training_history(history)
    config.PHASE1_EPOCHS = original_phase1_epochs
    config.PHASE2_EPOCHS = original_phase2_epochs
    print("\n✓ Training pipeline smoke test completed successfully!")
