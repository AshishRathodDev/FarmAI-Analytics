# test_setup.py - Create this file in root directory
import sys
from pathlib import Path

print("="*70)
print("TESTING PROJECT SETUP")
print("="*70)

# Test 1: Check project structure
project_root = Path(__file__).parent
print(f"\n1. Project root: {project_root}")
print(f"   Exists: {project_root.exists()}")

# Test 2: Check src directory
src_dir = project_root / "src"
print(f"\n2. Source directory: {src_dir}")
print(f"   Exists: {src_dir.exists()}")

# Test 3: Check config file
config_file = src_dir / "config.py"
print(f"\n3. Config file: {config_file}")
print(f"   Exists: {config_file.exists()}")

# Test 4: Try importing
sys.path.insert(0, str(project_root))
try:
    from src import config
    print("\n4.  Config imported successfully!")
    print(f"   RAW_DATA_DIR: {config.RAW_DATA_DIR}")
    print(f"   MODELS_DIR: {config.MODELS_DIR}")
except Exception as e:
    print(f"\n4.  Import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)