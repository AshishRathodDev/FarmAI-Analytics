"""
Main script to run the complete training pipeline for Syngenta Crop Disease Classification

Handles data preparation, model training, and saving artifacts.
"""

import sys
from pathlib import Path
import json

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import config
from data_utils import DataPipeline, load_class_indices
from train import train_model_pipeline, plot_training_history, save_training_history

def print_training_summary(train_gen, val_gen, num_classes, model_summary_str):
    """Prints a one-page summary of training configuration before starting."""
    print("\n================================================================================")
    print(" " * 25 + "TRAINING CONFIGURATION SUMMARY")
    print("================================================================================")
    print(f"\nDataset Path: {config.RAW_DATA_DIR}")
    print(f"Number of Classes: {num_classes}")
    print(f"Train Samples: {train_gen.samples}")
    print(f"Validation Samples: {val_gen.samples}")
    print(f"Image Size: {config.IMG_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    
    print("\nModel Summary (Partial):")
    # Truncate model summary if too long, as it will be in the actual model.summary()
    print("\n".join(model_summary_str.split('\n')[:15])) # Print first 15 lines of summary
    print("...") # Indicate truncation
    
    print("\nTraining Hyperparameters:")
    print(f"  Phase 1 Epochs (Frozen Base): {config.PHASE1_EPOCHS}")
    print(f"  Phase 1 Learning Rate: {config.PHASE1_LEARNING_RATE}")
    print(f"  Phase 2 Epochs (Fine-tune): {config.PHASE2_EPOCHS}")
    print(f"  Phase 2 Learning Rate: {config.PHASE2_LEARNING_RATE}")
    print(f"  Unfreeze Layers for Fine-tuning: {config.UNFREEZE_LAYERS}")
    print(f"  Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"  ReduceLROnPlateau Patience: {config.REDUCE_LR_PATIENCE}")
    print(f"  Random Seed: {config.RANDOM_SEED}")
    print("\n================================================================================")

def main():
    """Main function to execute the full training pipeline."""
    print("\n================================================================================")
    print(" " * 20 + "SYNGENTA CROP DISEASE CLASSIFIER")
    print(" " * 25 + "TRAINING PIPELINE EXECUTION")
    print("================================================================================")

    try:
        config.validate_config()
        config.get_config_summary()

        # --- Phase 1: Data Preparation ---
        print("\n[PHASE 1] Preparing data generators...")
        data_pipeline = DataPipeline()
        data_pipeline.verify_dataset_structure()
        train_generator, val_generator, test_generator, class_indices = data_pipeline.create_data_generators()
        num_classes = len(class_indices)
        
        # Save class indices
        with open(config.CLASS_INDICES_PATH, 'w') as f:
            json.dump(class_indices, f, indent=4)
        print(f"✓ Class indices saved to: {config.CLASS_INDICES_PATH}")

        # Get a dummy model summary for pre-training print
        from src.model import build_transfer_learning_model
        dummy_model_for_summary = build_transfer_learning_model(num_classes)
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            dummy_model_for_summary.summary()
        model_summary_str = f.getvalue()
        del dummy_model_for_summary # Clean up dummy model

        print_training_summary(train_generator, val_generator, num_classes, model_summary_str)
        
        # --- Phase 2: Model Training ---
        print("\n[PHASE 2] Starting model training...")
        model, history = train_model_pipeline(train_generator, val_generator, num_classes)
        
        # --- Phase 3: Save and Plot Training Artifacts ---
        print("\n[PHASE 3] Saving training artifacts...")
        plot_training_history(history, save_path=config.TRAINING_CURVES_FIGURE)
        save_training_history(history, save_path=config.TRAINING_HISTORY_PATH)
        
        print("\n================================================================================")
        print(" " * 25 + "TRAINING PIPELINE COMPLETE")
        print("================================================================================")
        print(f"✓ Final trained model saved to: {config.FINAL_MODEL_PATH}")
        print(f"✓ Best model weights checkpoint: {config.CHECKPOINT_FILEPATH}")
        print(f"✓ Training history plot: {config.TRAINING_CURVES_FIGURE}")
        print(f"✓ Training history data: {config.TRAINING_HISTORY_PATH}")

        return True

    except FileNotFoundError as e:
        print(f"\n ERROR: {e}")
        print("DIAGNOSIS: A required file or directory for data or model loading was not found.")
        print("ACTION: Please ensure the dataset is correctly placed and previous setup scripts (setup_project.py, setup_data.py) have run successfully.")
        return False
    except Exception as e:
        print(f"\n An unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
    
    