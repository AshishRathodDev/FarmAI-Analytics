#!/bin/bash

# =============================================================================
# FarmAI Assistant - Complete Fix Script
# =============================================================================
# This script will fix all common issues in your FarmAI project
# Run with: ./fix_farmai.sh
# =============================================================================

echo "============================================================"
echo "🌾 FarmAI Assistant - Automated Fix Script"
echo "============================================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# STEP 1: Check if virtual environment is activated
# =============================================================================
echo "🔍 Step 1: Checking Virtual Environment..."
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${RED}❌ Virtual environment not activated!${NC}"
    echo -e "${YELLOW}💡 Please run: source venv/bin/activate${NC}"
    echo ""
    exit 1
else
    echo -e "${GREEN}✅ Virtual environment is active: $VIRTUAL_ENV${NC}"
fi
echo ""

# =============================================================================
# STEP 2: Install Missing Dependencies
# =============================================================================
echo "============================================================"
echo "📦 Step 2: Installing Missing Dependencies..."
echo "============================================================"

echo "Installing python-dotenv..."
pip install python-dotenv --quiet

echo "Installing tensorflow (this may take a few minutes)..."
pip install tensorflow --quiet

echo "Installing other required packages..."
pip install numpy pandas matplotlib seaborn Pillow --quiet

echo "Installing additional packages for API..."
pip install fastapi uvicorn requests python-multipart --quiet

echo -e "${GREEN}✅ All dependencies installed!${NC}"
echo ""

# =============================================================================
# STEP 3: Create Missing Directories
# =============================================================================
echo "============================================================"
echo "📁 Step 3: Creating Missing Directories..."
echo "============================================================"

# Create directory function
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo -e "${GREEN}✅ Created: $1${NC}"
    else
        echo -e "${YELLOW}ℹ️  Already exists: $1${NC}"
    fi
}

# Create all required directories
create_dir "data/raw/plantvillage/color"
create_dir "data/processed"
create_dir "outputs/figures"
create_dir "outputs/metrics"
create_dir "outputs/models"
create_dir "models"
create_dir "logs"
create_dir "uploads"

echo ""

# =============================================================================
# STEP 4: Check Dataset
# =============================================================================
echo "============================================================"
echo "🌿 Step 4: Checking Dataset..."
echo "============================================================"

DATA_DIR="data/raw/plantvillage/color"
if [ -d "$DATA_DIR" ]; then
    CLASS_COUNT=$(find "$DATA_DIR" -maxdepth 1 -type d | wc -l)
    CLASS_COUNT=$((CLASS_COUNT - 1))  # Subtract parent directory
    
    if [ $CLASS_COUNT -gt 0 ]; then
        echo -e "${GREEN}✅ Dataset found: $CLASS_COUNT class directories${NC}"
        
        # Show first 5 classes
        echo "Sample classes:"
        ls "$DATA_DIR" | head -5 | while read class; do
            echo "  - $class"
        done
    else
        echo -e "${RED}❌ No class directories found in $DATA_DIR${NC}"
        echo -e "${YELLOW}💡 Please download PlantVillage dataset!${NC}"
    fi
else
    echo -e "${RED}❌ Dataset directory not found!${NC}"
    echo -e "${YELLOW}💡 Please download PlantVillage dataset to: $DATA_DIR${NC}"
fi
echo ""

# =============================================================================
# STEP 5: Verify Configuration Files
# =============================================================================
echo "============================================================"
echo "⚙️  Step 5: Verifying Configuration Files..."
echo "============================================================"

# Check .env file
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ .env file exists${NC}"
    ENV_LINES=$(grep -v '^#' .env | grep -v '^$' | wc -l)
    echo "   Found $ENV_LINES environment variables"
else
    echo -e "${YELLOW}⚠️  .env file not found. Creating template...${NC}"
    cat > .env << 'EOF'
# FarmAI Assistant Configuration
HUGGINGFACE_TOKEN=your_token_here
MODEL_PATH=models/efficientnet_model.h5
API_PORT=8000
EOF
    echo -e "${GREEN}✅ Created .env template${NC}"
    echo -e "${YELLOW}💡 Please update .env with your actual values!${NC}"
fi

# Check config.py
if [ -f "src/config.py" ]; then
    echo -e "${GREEN}✅ src/config.py exists${NC}"
else
    echo -e "${RED}❌ src/config.py not found!${NC}"
fi

# Check requirements.txt
if [ -f "requirements.txt" ]; then
    echo -e "${GREEN}✅ requirements.txt exists${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt not found. Creating...${NC}"
    cat > requirements.txt << 'EOF'
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
Pillow>=8.0.0
tensorflow>=2.10.0
python-dotenv>=0.19.0
fastapi>=0.95.0
uvicorn>=0.21.0
python-multipart>=0.0.6
requests>=2.28.0
EOF
    echo -e "${GREEN}✅ Created requirements.txt${NC}"
fi
echo ""

# =============================================================================
# STEP 6: Check Model Files
# =============================================================================
echo "============================================================"
echo "🤖 Step 6: Checking Model Files..."
echo "============================================================"

MODEL_FOUND=false

# Check common model locations
if [ -f "models/efficientnet_model.h5" ]; then
    SIZE=$(du -h "models/efficientnet_model.h5" | cut -f1)
    echo -e "${GREEN}✅ Model found: models/efficientnet_model.h5 ($SIZE)${NC}"
    MODEL_FOUND=true
elif [ -f "models/efficientnet_model.keras" ]; then
    SIZE=$(du -h "models/efficientnet_model.keras" | cut -f1)
    echo -e "${GREEN}✅ Model found: models/efficientnet_model.keras ($SIZE)${NC}"
    MODEL_FOUND=true
elif [ -f "outputs/models/efficientnet_best_model.h5" ]; then
    SIZE=$(du -h "outputs/models/efficientnet_best_model.h5" | cut -f1)
    echo -e "${GREEN}✅ Model found: outputs/models/efficientnet_best_model.h5 ($SIZE)${NC}"
    MODEL_FOUND=true
fi

if [ "$MODEL_FOUND" = false ]; then
    echo -e "${YELLOW}⚠️  No trained model found${NC}"
    echo -e "${YELLOW}💡 You need to train the model first!${NC}"
    echo "   Run: jupyter notebook notebooks/"
fi
echo ""

# =============================================================================
# STEP 7: Create Helper Scripts
# =============================================================================
echo "============================================================"
echo "📝 Step 7: Creating Helper Scripts..."
echo "============================================================"

# Create start_backend.sh
cat > start_backend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting FarmAI Backend..."
source venv/bin/activate

# Check which API file exists
if [ -f "api.py" ]; then
    python api.py
elif [ -f "app.py" ]; then
    python app.py
elif [ -f "src/api/main.py" ]; then
    python src/api/main.py
else
    echo "❌ No API file found!"
    exit 1
fi
EOF
chmod +x start_backend.sh
echo -e "${GREEN}✅ Created start_backend.sh${NC}"

# Create start_jupyter.sh
cat > start_jupyter.sh << 'EOF'
#!/bin/bash
echo "📓 Starting Jupyter Notebook..."
source venv/bin/activate
jupyter notebook notebooks/
EOF
chmod +x start_jupyter.sh
echo -e "${GREEN}✅ Created start_jupyter.sh${NC}"

# Create run_diagnostic.sh
cat > run_diagnostic.sh << 'EOF'
#!/bin/bash
echo "🔍 Running System Diagnostic..."
source venv/bin/activate
python diagnostic.py
EOF
chmod +x run_diagnostic.sh
echo -e "${GREEN}✅ Created run_diagnostic.sh${NC}"

echo ""

# =============================================================================
# STEP 8: Test Imports
# =============================================================================
echo "============================================================"
echo "🧪 Step 8: Testing Python Imports..."
echo "============================================================"

python << 'PYTHON_EOF'
import sys

packages = [
    ('dotenv', 'python-dotenv'),
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn'),
    ('PIL', 'Pillow'),
    ('tensorflow', 'tensorflow'),
]

all_ok = True
for module, package in packages:
    try:
        __import__(module)
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package} - FAILED")
        all_ok = False

sys.exit(0 if all_ok else 1)
PYTHON_EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All imports successful!${NC}"
else
    echo -e "${RED}❌ Some imports failed!${NC}"
    echo -e "${YELLOW}💡 Try: pip install -r requirements.txt${NC}"
fi
echo ""

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo "============================================================"
echo "�� SETUP SUMMARY"
echo "============================================================"
echo ""
echo "✅ Dependencies installed"
echo "✅ Directories created"
echo "✅ Configuration checked"
echo "✅ Helper scripts created"
echo ""
echo "============================================================"
echo "�� NEXT STEPS"
echo "============================================================"
echo ""
echo "1️⃣  Check if dataset exists:"
echo "   ls data/raw/plantvillage/color/"
echo ""
echo "2️⃣  Train model (if not already trained):"
echo "   ./start_jupyter.sh"
echo "   Then run: 00_data_inspection_and_eda.ipynb"
echo ""
echo "3️⃣  Start backend server:"
echo "   ./start_backend.sh"
echo ""
echo "4️⃣  Run diagnostic anytime:"
echo "   ./run_diagnostic.sh"
echo ""
echo "============================================================"
echo -e "${GREEN}✅ FarmAI Setup Complete!${NC}"
echo "============================================================"
echo ""
