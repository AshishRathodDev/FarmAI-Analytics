from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Allow React to connect

# Health check
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "online", "message": "FarmAI API is running"})

# Chat endpoint - Real responses
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    # Simple rule-based responses (you can add Gemini API later)
    responses = {
        'disease': "Common crop diseases include Late Blight, Early Blight, and Bacterial Spot. Upload an image for detailed diagnosis!",
        'fertilizer': "For tomatoes, use NPK 10-10-10 fertilizer every 2 weeks. Ensure soil pH is 6.0-6.8.",
        'pest': "Use neem oil spray for general pest control. For specific pests, upload a photo!",
        'default': f"I understand you're asking about '{user_message}'. For accurate disease detection, please use our Scanner feature to upload a plant image!"
    }
    
    # Match keywords
    response_text = responses['default']
    if 'disease' in user_message.lower():
        response_text = responses['disease']
    elif 'fertilizer' in user_message.lower():
        response_text = responses['fertilizer']
    elif 'pest' in user_message.lower():
        response_text = responses['pest']
    
    return jsonify({
        "success": True,
        "response": response_text,
        "timestamp": "now"
    })

# Image prediction endpoint
@app.route('/api/predict', methods=['POST'])
def predict():
    # This will connect to your TensorFlow model
    return jsonify({
        "success": True,
        "disease": "Tomato Late Blight",
        "confidence": 93.5,
        "severity": "high",
        "treatment": "Apply Mancozeb fungicide (2g/L) within 24 hours. Remove affected leaves."
    })

if __name__ == '__main__':
    print("🚀 FarmAI API Server Starting...")
    print("📡 Backend running on http://localhost:5050")
    app.run(debug=True, port=5050, host='0.0.0.0')



