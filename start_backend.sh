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
