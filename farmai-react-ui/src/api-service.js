// FarmAI API Service
const API_BASE_URL = 'http://localhost:5050/api';

export async function predictDisease(imageFile) {
  try {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (data.status === 'success') {
      return {
        status: 'success',
        disease: data.prediction,
        confidence_percentage: Math.round(data.confidence * 100),
        recommendation: `Treatment: Remove infected leaves, apply fungicide, improve air circulation.`,
        top_3: data.top_3
      };
    } else {
      return {
        status: 'error',
        message: data.message || 'Prediction failed'
      };
    }
  } catch (error) {
    console.error('API Error:', error);
    return {
      status: 'error',
      message: 'Failed to connect to server. Ensure Flask API is running on port 5050.'
    };
  }
}

export async function chatWithAI(message) {
  try {
    const responses = {
      'common crop diseases': 'Common diseases: Late Blight, Early Blight, Bacterial Spot, Septoria Leaf Spot, Powdery Mildew.',
      'fertilizer recommendations': 'Use balanced NPK (10-10-10). Organic: compost, cow dung, vermicompost.',
      'pest control methods': 'Use neem oil, companion planting, crop rotation, and biological controls.'
    };

    const lowerMessage = message.toLowerCase();
    for (const [key, response] of Object.entries(responses)) {
      if (lowerMessage.includes(key)) return response;
    }

    return 'For crop disease detection, use the Scanner page. Upload an image for specific diagnosis.';
  } catch (error) {
    return 'Sorry, I encountered an error. Please try again.';
  }
}
