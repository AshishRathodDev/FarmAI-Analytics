"""
Debug script to verify dataset structure

"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import config

def debug_dataset():
    print("\n" + "="*70)
    print("DATASET STRUCTURE DEBUG")
    print("="*70)
    
    print(f"\n1. Config Path: {config.RAW_DATA_DIR}")
    print(f"   Exists: {config.RAW_DATA_DIR.exists()}")
    
    if config.RAW_DATA_DIR.exists():
        print(f"\n2. Contents of {config.RAW_DATA_DIR}:")
        contents = list(config.RAW_DATA_DIR.iterdir())
        print(f"   Total items: {len(contents)}")
        
        # Show first 10 items
        for i, item in enumerate(contents[:10]):
            item_type = "DIR" if item.is_dir() else "FILE"
            print(f"   [{item_type}] {item.name}")
        
        if len(contents) > 10:
            print(f"   ... and {len(contents) - 10} more")
        
        # Count images in first directory
        if contents and contents[0].is_dir():
            first_class = contents[0]
            images = list(first_class.glob('*.[jJ][pP][gG]')) + \
                    list(first_class.glob('*.[pP][nN][gG]'))
            print(f"\n3. Sample class '{first_class.name}': {len(images)} images")
    
    else:
        print("\n❌ ERROR: Dataset directory not found!")
        print("\nPossible locations to check:")
        
        # Check parent directory
        parent = config.RAW_DATA_DIR.parent
        if parent.exists():
            print(f"\nContents of {parent}:")
            for item in parent.iterdir():
                print(f"  - {item.name}")
        
        # Suggest solutions
        print("\n💡 SOLUTIONS:")
        print("1. Check if dataset is downloaded")
        print("2. Verify folder name (color vs Color vs plant_village)")
        print("3. Update config.py RAW_DATA_DIR path")

if __name__ == "__main__":
    debug_dataset()