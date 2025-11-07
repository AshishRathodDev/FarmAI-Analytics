"""
Script to prepare (split) the dataset for Syngenta Crop Disease Classification

Runs the data verification and deterministic splitting process.

"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import config
from data_utils import DataPipeline

def main():
    """Main function to setup data splits."""
    print("\n================================================================================")
    print(" " * 25 + "DATA PREPARATION SCRIPT")
    print("================================================================================")

    try:
        config.validate_config()
        config.get_config_summary()

        data_pipeline = DataPipeline()
        data_pipeline.verify_dataset_structure()
        
        # Create splits (force recreate if needed)
        train_dir, val_dir, test_dir = data_pipeline.create_deterministic_splits(force_recreate=False)
        
        print("\n================================================================================")
        print(" " * 25 + "DATA PREPARATION COMPLETE")
        print("================================================================================")
        print(f"✓ Training data available at: {train_dir}")
        print(f"✓ Validation data available at: {val_dir}")
        print(f"✓ Test data available at: {test_dir}")
        print(f"✓ Class indices saved to: {config.CLASS_INDICES_PATH}")
        
        return True

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("DIAGNOSIS: The raw dataset was not found at the expected path.")
        print("ACTION: Please ensure the 'color' folder from PlantVillage dataset is placed inside 'data/raw/plantvillage/'.")
        return False
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print("DIAGNOSIS: The dataset structure is not as expected or classes are missing.")
        print("ACTION: Verify the contents of 'data/raw/plantvillage/color/' to ensure it contains class directories.")
        return False
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)