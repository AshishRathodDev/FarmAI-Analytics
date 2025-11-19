const API_BASE = "http://localhost:5050/api";

// POST image to Flask predict API
export const predictDisease = async (imageFile) => {
  try {
    const formData = new FormData();
    formData.append('file', imageFile);  // ✅ CORRECT - matches Flask

    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
      // Don't set Content-Type header - browser will set it automatically with boundary
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    
    if (data.status === 'error') {
      throw new Error(data.message || 'Prediction failed');
    }
    
    return data;

  } catch (error) {
    console.error("Error calling predict API:", error);
    return { 
      status: "error", 
      message: error.message || "Failed to connect to API" 
    };
  }
};

// POST message to Flask chat API
export const chatWithAI = async (message, crop = "General", language = "English") => {
  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { 
        'Content-Type': 'application/json' 
      },
      body: JSON.stringify({ 
        message,      // ✅ CORRECT - matches Flask
        crop,
        language
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    
    if (data.status === 'error') {
      throw new Error(data.message || 'Chat failed');
    }
    
    return data.response || "Sorry, could not get response.";
    
  } catch (error) {
    console.error("Error calling chat API:", error);
    return "Failed to connect to AI Chat API: " + error.message;
  }
};

// Get analytics summary
export const getAnalyticsSummary = async () => {
  try {
    const response = await fetch(`${API_BASE}/analytics/summary`);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error fetching analytics:", error);
    return { status: "error", message: error.message };
  }
};

// Export all functions
export default {
  predictDisease,
  chatWithAI,
  getAnalyticsSummary
};