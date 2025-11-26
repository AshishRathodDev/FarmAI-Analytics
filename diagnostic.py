#!/usr/bin/env python3
"""
FarmAI Diagnostic Script
Checks all components and identifies issues
"""

import os
import sys
from pathlib import Path
import importlib.util

def check_color(status):
    """Return color code for status"""
    return "✅" if status else "❌"

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n🔍 Checking Dependencies...")
    required = [
        'dotenv', 'numpy', 'pandas', 'matplotlib', 
        'seaborn', 'PIL', 'tensorflow'
    ]
    
    results = {}
    for pkg in required:
        try:
            if pkg == 'PIL':
                importlib.import_module('PIL')
            elif pkg == 'dotenv':
                importlib.import_module('dotenv')
            else:
                importlib.import_module(pkg)
            results[pkg] = True
            print(f"  {check_color(True)} {pkg}")
        except ImportError:
            results[pkg] = False
            print(f"  {check_color(False)} {pkg} - MISSING!")
    
    return all(results.values())

def check_directory_structure():
    """Check if project directories exist"""
    print("\n📁 Checking Directory Structure...")
    
    required_dirs = [
        'data/raw/plantvillage/color',
        'data/processed',
        'outputs/figures',
        'outputs/metrics',
        'notebooks',
        'src'
    ]
    
    results = {}
    for dir_path in required_dirs:
        path = Path(dir_path)
        exists = path.exists()
        results[dir_path] = exists
        print(f"  {check_color(exists)} {dir_path}")
        
        # Show sample count for data directory
        if 'color' in dir_path and exists:
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            print(f"     ℹ️  Found {len(subdirs)} class directories")
    
    return all(results.values())

def check_config_file():
    """Check if config files exist and are valid"""
    print("\n⚙️  Checking Configuration Files...")
    
    files_to_check = [
        'src/config.py',
        '.env',
        'requirements.txt'
    ]
    
    results = {}
    for file_path in files_to_check:
        path = Path(file_path)
        exists = path.exists()
        results[file_path] = exists
        print(f"  {check_color(exists)} {file_path}")
        
        if file_path == '.env' and exists:
            with open(path) as f:
                lines = [l for l in f.readlines() if l.strip() and not l.startswith('#')]
                print(f"     ℹ️  Found {len(lines)} environment variables")
    
    return all(results.values())

def check_model_files():
    """Check if model files exist"""
    print("\n🤖 Checking Model Files...")
    
    model_locations = [
        'models/efficientnet_model.h5',
        'models/efficientnet_model.keras',
        'outputs/models'
    ]
    
    for loc in model_locations:
        path = Path(loc)
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  ✅ {loc} ({size_mb:.1f} MB)")
            else:
                files = list(path.glob('*'))
                print(f"  ✅ {loc} ({len(files)} files)")
        else:
            print(f"  ℹ️  {loc} - Not found (may need training)")

def check_backend_status():
    """Check if backend is accessible"""
    print("\n🌐 Checking Backend Status...")
    
    try:
        import requests
        backends = [
            'http://localhost:8000',
            'http://localhost:5000',
            'http://127.0.0.1:8000',
        ]
        
        for backend_url in backends:
            try:
                response = requests.get(f"{backend_url}/health", timeout=2)
                if response.status_code == 200:
                    print(f"  ✅ Backend running at {backend_url}")
                    return True
            except:
                print(f"  ❌ {backend_url} - Not responding")
        
        print("\n  ⚠️  No backend server detected!")
        print("  💡 Start backend with: python src/api/main.py")
        return False
        
    except ImportError:
        print("  ℹ️  'requests' package not installed - skipping backend check")
        return None

def generate_fix_script():
    """Generate a script to fix common issues"""
    print("\n📝 Generating Fix Script...")
    
    fix_script = """#!/bin/bash
# FarmAI Fix Script
# Run this to fix common issues

echo "🔧 Installing missing dependencies..."
pip install python-dotenv numpy pandas matplotlib seaborn Pillow tensorflow

echo "📁 Creating missing directories..."
mkdir -p data/raw/plantvillage/color
mkdir -p data/processed
mkdir -p outputs/figures
mkdir -p outputs/metrics
mkdir -p outputs/models
mkdir -p models

echo "✅ Basic setup complete!"
echo ""
echo "⚠️  Next steps:"
echo "1. Download PlantVillage dataset to data/raw/plantvillage/color/"
echo "2. Create .env file with necessary variables"
echo "3. Run the notebook: jupyter notebook notebooks/00_data_inspection_and_eda.ipynb"
echo "4. Start backend: python src/api/main.py"
"""
    
    with open('fix_farmai.sh', 'w') as f:
        f.write(fix_script)
    
    os.chmod('fix_farmai.sh', 0o755)
    print("  ✅ Created fix_farmai.sh")
    print("  💡 Run with: ./fix_farmai.sh")

def main():
    """Run all diagnostic checks"""
    print("=" * 60)
    print("🌾 FarmAI Assistant - System Diagnostic")
    print("=" * 60)
    
    # Run all checks
    deps_ok = check_dependencies()
    dirs_ok = check_directory_structure()
    config_ok = check_config_file()
    check_model_files()
    backend_ok = check_backend_status()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    all_critical_ok = deps_ok and dirs_ok and config_ok
    
    if all_critical_ok and backend_ok:
        print("✅ All systems operational!")
    elif all_critical_ok:
        print("⚠️  Core components OK, but backend needs attention")
    else:
        print("❌ Critical issues found - see above for details")
        print("\n💡 Recommended Actions:")
        if not deps_ok:
            print("   1. Install missing dependencies: pip install -r requirements.txt")
        if not dirs_ok:
            print("   2. Create missing directories or download dataset")
        if not config_ok:
            print("   3. Set up configuration files (.env, config.py)")
        if not backend_ok:
            print("   4. Start backend server: python src/api/main.py")
    
    # Generate fix script
    generate_fix_script()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
