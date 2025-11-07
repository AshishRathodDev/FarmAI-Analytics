"""
Script to run model evaluation for Syngenta Crop Disease Classification

Loads a trained model and performs comprehensive evaluation.

"""

import sys
from pathlib import Path
import tensorflow as tf

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import config
from data_utils import DataPipeline, load_class_indices
from evaluate import evaluate_model_performance, plot_confusion_matrix, visualize_sample_predictions, visualize_gradcam_for_samples

def main():
    """Main function to execute the full evaluation pipeline."""
    print("\n================================================================================")
    print(" " * 25 + "MODEL EVALUATION SCRIPT")
    print("================================================================================")

    try:
        config.validate_config()
        config.get_config_summary()

        # --- Phase 1: Load Model and Class Indices ---
        print("\n[PHASE 1] Loading trained model and class indices...")
        if not config.FINAL_MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found at: {config.FINAL_MODEL_PATH}")
        model = tf.keras.models.load_model(config.FINAL_MODEL_PATH)
        print(f"✓ Model loaded from: {config.FINAL_MODEL_PATH}")

        class_indices = load_class_indices()
        class_names = list(class_indices.keys())
        print(f"✓ Class indices loaded. Total classes: {len(class_names)}")

        # --- Phase 2: Prepare Test Data Generator ---
        print("\n[PHASE 2] Preparing test data generator...")
        data_pipeline = DataPipeline()
        data_pipeline.verify_dataset_structure()
        _, _, test_generator, _ = data_pipeline.create_data_generators()
        
        # Ensure test_generator's class_indices match the loaded ones
        test_generator.class_indices = class_indices
        test_generator.class_names = class_names

        # --- Phase 3: Perform Evaluation ---
        print("\n[PHASE 3] Performing comprehensive model evaluation...")
        accuracy, predicted_classes, true_labels, report_df = evaluate_model_performance(
            model, test_generator, class_names
        )

        # --- Phase 4: Generate Visualizations ---
        print("\n[PHASE 4] Generating evaluation visualizations...")
        plot_confusion_matrix(true_labels, predicted_classes, class_names, save_path=config.CONFUSION_MATRIX_FIGURE)
        visualize_sample_predictions(model, test_generator, class_names, save_path=config.PREDICTIONS_FIGURE)
        visualize_gradcam_for_samples(model, test_generator, class_names, save_path=config.GRADCAM_FIGURE)

        print("\n================================================================================")
        print(" " * 20 + "MODEL EVALUATION COMPLETE")
        print("================================================================================")
        print(f"✓ Overall Test Accuracy: {accuracy*100:.2f}%")
        print(f"✓ Classification Report saved to: {config.CLASSIFICATION_REPORT_PATH}")
        print(f"✓ Confusion Matrix plot: {config.CONFUSION_MATRIX_FIGURE}")
        print(f"✓ Sample Predictions plot: {config.PREDICTIONS_FIGURE}")
        print(f"✓ Grad-CAM visualizations: {config.GRADCAM_FIGURE}")

        return True

    except FileNotFoundError as e:
        print(f"\n ERROR: {e}")
        print("DIAGNOSIS: The trained model or class indices were not found. Ensure training completed successfully.")
        print("ACTION: Run 'python scripts/run_training.py' first to train and save the model.")
        return False
    except Exception as e:
        print(f"\n An unexpected error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
    
    