// Hardcode API URL (temporary fix)
const API_BASE_URL = 'https://farmai-analytics.onrender.com';

console.log('🔧 API Base URL:', API_BASE_URL);

export async function predictDisease(imageFile) {
  try {
    const formData = new FormData();
    formData.append('file', imageFile);

    console.log('📤 Sending to:', `${API_BASE_URL}/api/predict`);

    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    if (data.status === 'success') {
      return {
        status: 'success',
        disease: data.prediction,
        confidence_percentage: Math.round(data.confidence * 100),
        recommendation: data.recommendation || 'Apply recommended fungicide treatment',
        top_3: data.top_3 || []
      };
    } else {
      return {
        status: 'error',
        message: data.message || 'Prediction failed'
      };
    }
  } catch (error) {
    console.error('❌ API Error:', error);
    return {
      status: 'error',
      message: `Failed: ${error.message}`
    };
  }
}

export async function chatWithAI(message) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });
    const data = await response.json();
    return data.success ? data.response : 'Error';
  } catch (error) {
    console.error('Chat error:', error);
    return 'Unable to connect';
  }
}
