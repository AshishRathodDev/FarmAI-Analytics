// FarmAI API Service

// Best Practice: Use .env for production/development
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5050/api';

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
        recommendation: data.recommendation || data.treatment || 'No treatment data available',
        top_3: data.top_3 || []
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
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });

    const data = await response.json();

    if (data.success) {
      return data.response;
    } else {
      return 'Sorry, I encountered an error. Please try again.';
    }
  } catch (error) {
    console.error('Chat API Error:', error);
    return 'Unable to connect to AI assistant. Please check your connection.';
  }
}

