/**
 * FarmAI API Service - DIRECT MODE (No Timeout)
 * This module handles communication with the FarmAI backend server     
 */

const API_BASE_URL = 'https://farmai-backend-148791329286.asia-south1.run.app';

// Logging helper
const log = (...args) => console.log('%c🌾 FarmAI', 'color: #10B981; font-weight: bold', ...args);

// ============================================================================
// 1. CHAT FUNCTION (Frontend Only)
// ============================================================================
export async function chatWithAI(message) {
  await new Promise(r => setTimeout(r, 3000)); 
  const msg = message.toLowerCase();
  
  if (msg.includes('hello') || msg.includes('hi')) return "Namaste! I am FarmAI. Upload a leaf photo in Scan page.";
  if (msg.includes('tomato') || msg.includes('potato')) return "I can detect Early Blight, Late Blight, and Viruses in Tomato & Potato.";
  if (msg.includes('medicine') || msg.includes('cure')) return "Please upload a photo first so I can suggest the right medicine.";
  
  return "I am trained to help farmers. Please go to the Scan page to check your crop health.";
}

// ============================================================================
// 2. PREDICT FUNCTION (Main Logic - No Timeout)
// ============================================================================
export async function predictDisease(imageFile) {
  log('🚀 STARTING PREDICTION (Direct Mode - No Timeout)...');
  
  try {
    // Step A: Validate
    if (!imageFile) throw new Error('No image file provided');

    // Step B: Prepare Data
    const formData = new FormData();
    formData.append('file', imageFile);

    log('⏳ Sending request to Server... (Please wait, do not refresh)');
    
    
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      mode: 'cors', 
      body: formData
    });

    log('✅ Response Status:', response.status);

    // Step C: Check Errors
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server Error: ${response.status} - ${errorText}`);
    }

    // Step D: Get Data
    const data = await response.json();
    log('🎉 Result Received:', data);

    return {
      status: 'success',
      disease: data.prediction,
      confidence: data.confidence,
      confidence_percentage: data.confidence_percent,
      recommendation: getRecommendation(data.prediction),
      top_3: data.top_3 || []
    };

  } catch (error) {
    console.error("Prediction Error:", error);
    return {
      status: 'error',
      message: "Connection failed. The server is waking up. Please try again in 30 seconds."
    };
  }
}

// Helper for recommendations
function getRecommendation(disease) {
  if (!disease) return "Consult a local expert.";
  const d = disease.toLowerCase();
  
  if (d.includes('early blight')) return "Use fungicides like Chlorothalonil or Mancozeb. Improve air circulation.";
  if (d.includes('late blight')) return "Apply copper-based fungicides immediately. Destroy infected plants.";
  if (d.includes('bacterial')) return "Use copper sprays. Avoid overhead watering to keep leaves dry.";
  if (d.includes('virus') || d.includes('yellow')) return "Remove infected plants to stop spread. Control whiteflies.";
  if (d.includes('healthy')) return "Your crop looks healthy! Keep monitoring regularly.";
  
  return "Ensure proper sunlight, water, and nutrition for your crops.";
}

export const API_CONFIG = { version: 'DIRECT.MODE' };



