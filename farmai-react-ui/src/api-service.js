/**
 * FarmAI API Service - Complete with Backend Wake-Up Logic
 * Backend: https://farmai-analytics.onrender.com
 */

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL = 'https://farmai-analytics.onrender.com';

// Debug mode
const DEBUG = true;

const log = (...args) => {
  if (DEBUG) console.log('🌾 [FarmAI API]', ...args);
};

const logError = (...args) => {
  console.error('❌ [FarmAI API]', ...args);
};

// ============================================================================
// Health Check
// ============================================================================

export async function checkBackendHealth() {
  try {
    log('Checking backend health...');
    
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }

    const data = await response.json();
    log('✅ Backend healthy:', data);
    
    return {
      success: true,
      isHealthy: data.status === 'healthy',
      modelLoaded: data.model_loaded,
      classesCount: data.classes_count || 0,
    };
  } catch (error) {
    logError('Health check failed:', error);
    return {
      success: false,
      isHealthy: false,
      error: error.message,
    };
  }
}

// ============================================================================
// Disease Prediction with Wake-Up Logic
// ============================================================================

export async function predictDisease(imageFile) {
  try {
    // Validate input
    if (!imageFile) {
      throw new Error('No image file provided');
    }

    log('📤 Starting prediction...');
    log('File:', imageFile.name, 'Size:', imageFile.size, 'Type:', imageFile.type);

    // STEP 1: Wake up backend first
    log('⏰ Waking up backend (this may take 50-60 seconds on first request)...');
    
    let backendReady = false;
    const maxWakeAttempts = 3;
    
    for (let i = 0; i < maxWakeAttempts; i++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 sec per attempt
        
        const healthCheck = await fetch(`${API_BASE_URL}/health`, {
          method: 'GET',
          signal: controller.signal,
          headers: {
            'Accept': 'application/json',
          },
        });
        
        clearTimeout(timeoutId);
        
        if (healthCheck.ok) {
          const data = await healthCheck.json();
          if (data.status === 'healthy' && data.model_loaded) {
            log('✅ Backend is awake and ready!');
            backendReady = true;
            break;
          } else {
            log('⚠️ Backend responding but model not loaded yet...');
          }
        }
      } catch (healthError) {
        if (healthError.name === 'AbortError') {
          log(`⚠️ Wake attempt ${i + 1}/${maxWakeAttempts} timed out, retrying...`);
        } else {
          log(`⚠️ Wake attempt ${i + 1}/${maxWakeAttempts} failed:`, healthError.message);
        }
        
        // Wait 5 seconds before retry
        if (i < maxWakeAttempts - 1) {
          await new Promise(resolve => setTimeout(resolve, 5000));
        }
      }
    }

    if (!backendReady) {
      log('⚠️ Backend not responding to health checks, attempting prediction anyway...');
    }

    // STEP 2: Now send prediction request
    const formData = new FormData();
    formData.append('file', imageFile);

    log('📤 Sending prediction request to:', `${API_BASE_URL}/api/predict`);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 min timeout

    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
      // Don't set Content-Type - browser will set it with boundary
      headers: {
        'Accept': 'application/json',
      },
    });

    clearTimeout(timeoutId);

    log('Response status:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      logError('API Error Response:', errorText);
      throw new Error(`API returned ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    log('✅ Prediction response:', data);

    if (data.status === 'success') {
      return {
        status: 'success',
        disease: data.prediction || 'Unknown Disease',
        confidence: data.confidence || 0,
        confidence_percentage: Math.round((data.confidence || 0) * 100),
        confidence_percent: data.confidence_percent || '0%',
        recommendation: getRecommendation(data.prediction),
        top_3: data.top_3 || [],
      };
    } else {
      throw new Error(data.message || 'Prediction failed');
    }

  } catch (error) {
    if (error.name === 'AbortError') {
      logError('Request timeout - backend cold start taking too long');
      return {
        status: 'error',
        message: '⏰ Backend is waking up from sleep (this takes 50-60 seconds on free tier). Please wait 1 minute and try again.',
      };
    }

    logError('Prediction error:', error);
    return {
      status: 'error',
      message: error.message || 'Failed to predict disease',
    };
  }
}

// ============================================================================
// Chat with AI
// ============================================================================

export async function chatWithAI(message) {
  try {
    log('💬 Sending chat message:', message);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 sec timeout

    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ message }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Chat API returned ${response.status}`);
    }

    const data = await response.json();
    log('✅ Chat response:', data);

    return data.response || data.message || 'No response';

  } catch (error) {
    logError('Chat error:', error);
    
    // Return fallback response
    return 'The AI chat feature is currently unavailable. Please use the disease scanner for plant diagnosis.';
  }
}

// ============================================================================
// Get Disease Classes
// ============================================================================

export async function getDiseaseClasses() {
  try {
    log('Fetching disease classes...');

    const response = await fetch(`${API_BASE_URL}/api/classes`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Classes API returned ${response.status}`);
    }

    const data = await response.json();
    log('✅ Classes loaded:', data.count);

    return {
      success: true,
      classes: data.classes || [],
      count: data.count || 0,
    };

  } catch (error) {
    logError('Error fetching classes:', error);
    return {
      success: false,
      error: error.message,
    };
  }
}

// ============================================================================
// Helper: Get Treatment Recommendation
// ============================================================================

function getRecommendation(diseaseName) {
  const recommendations = {
    'Early Blight': 'Apply copper-based fungicide. Remove infected leaves. Ensure proper spacing for air circulation.',
    'Late Blight': 'Apply fungicide immediately. Remove and destroy infected plants. Avoid overhead watering.',
    'Leaf Mold': 'Improve air circulation. Reduce humidity. Apply fungicide if severe.',
    'Septoria Leaf Spot': 'Remove infected leaves. Apply copper fungicide. Mulch to prevent soil splash.',
    'Bacterial Spot': 'Use copper-based bactericide. Remove infected tissue. Avoid overhead irrigation.',
    'Target Spot': 'Apply fungicide. Ensure good air circulation. Remove plant debris.',
    'Mosaic Virus': 'Remove and destroy infected plants. Control aphids. Use resistant varieties.',
    'Yellow Leaf Curl': 'Control whiteflies. Remove infected plants. Use resistant varieties.',
    'Healthy': 'No treatment needed. Continue regular monitoring and good cultural practices.',
  };

  // Try to match disease name
  for (const [key, value] of Object.entries(recommendations)) {
    if (diseaseName && diseaseName.toLowerCase().includes(key.toLowerCase())) {
      return value;
    }
  }

  return 'Consult with local agricultural extension service for specific treatment recommendations.';
}

// ============================================================================
// Export API Configuration (for debugging)
// ============================================================================

export const API_CONFIG = {
  baseUrl: API_BASE_URL,
  endpoints: {
    health: `${API_BASE_URL}/health`,
    predict: `${API_BASE_URL}/api/predict`,
    chat: `${API_BASE_URL}/api/chat`,
    classes: `${API_BASE_URL}/api/classes`,
  },
  timeout: {
    health: 60000, // 60 seconds
    predict: 180000, // 3 minutes (for cold start)
    chat: 30000, // 30 seconds
  },
  debug: DEBUG,
};

// Log configuration on module load
log('API Configuration:', API_CONFIG);
log('⚠️ Note: First request after 15 min of inactivity will take 50-60 seconds (cold start)');
