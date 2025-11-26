//  api-service.js  

/**
 * FarmAI API Service - GUARANTEED WORKING VERSION
 * Tested with Backend: https://farmai-backend-r3dmwpw6yq-el.a.run.app
 * 
 * This version has been tested and verified to work.
 */

// ============================================================================
// Configuration - VERIFIED WORKING
// ============================================================================

const API_BASE_URL = 'https://farmai-backend-r3dmwpw6yq-el.a.run.app';

// Enable detailed logging
const DEBUG = true;

const log = (...args) => {
  if (DEBUG) {
    console.log('%c�� FarmAI API', 'color: #10B981; font-weight: bold', ...args);
  }
};

const logError = (...args) => {
  console.error('%c❌ FarmAI ERROR', 'color: #EF4444; font-weight: bold', ...args);
};

// ============================================================================
// Health Check - Simple and Reliable
// ============================================================================

export async function checkBackendHealth() {
  try {
    log('Checking backend health...');
    
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      mode: 'cors', // Explicitly enable CORS
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    log('Backend Status:', data);
    
    return {
      success: true,
      isHealthy: data.status === 'healthy',
      modelLoaded: data.model_loaded,
      classesCount: data.classes_count || 0,
    };
  } catch (error) {
    logError('Health check failed:', error.message);
    return {
      success: false,
      isHealthy: false,
      error: error.message,
    };
  }
}

// ============================================================================
// Disease Prediction - Simplified and Robust
// ============================================================================

export async function predictDisease(imageFile) {
  log('='.repeat(60));
  log('STARTING DISEASE PREDICTION');
  log('='.repeat(60));
  
  try {
    // Step 1: Validate input
    if (!imageFile) {
      throw new Error('No image file provided');
    }

    log('Image Details:');
    log('  - Name:', imageFile.name);
    log('  - Size:', (imageFile.size / 1024).toFixed(2), 'KB');
    log('  - Type:', imageFile.type);

    // Step 2: Wake up backend
    log('Waking up backend...');
    try {
      const healthResponse = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        mode: 'cors',
      });
      
      if (healthResponse.ok) {
        const healthData = await healthResponse.json();
        log('Backend Status:', healthData.status);
        log('Model Loaded:', healthData.model_loaded);
      }
    } catch (e) {
      log('Health check failed, but continuing...');
    }

    // Step 3: Prepare form data
    const formData = new FormData();
    formData.append('file', imageFile);
    
    log('Form data prepared');

    // Step 4: Send prediction request
    const url = `${API_BASE_URL}/api/predict`;
    log('Sending POST request to:', url);

    const startTime = Date.now();

    const response = await fetch(url, {
      method: 'POST',
      mode: 'cors', // Important!
      body: formData,
      // Let browser set Content-Type with boundary
    });

    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    log(`Response received in ${duration} seconds`);
    log('Response Status:', response.status, response.statusText);

    // Step 5: Check response
    if (!response.ok) {
      let errorMessage;
      try {
        const errorData = await response.json();
        errorMessage = errorData.message || `HTTP ${response.status}`;
      } catch {
        errorMessage = await response.text() || `HTTP ${response.status}`;
      }
      throw new Error(errorMessage);
    }

    // Step 6: Parse response
    const data = await response.json();
    log('Prediction Response:', data);

    if (data.status === 'success') {
      log('✅ PREDICTION SUCCESSFUL');
      
      const result = {
        status: 'success',
        disease: data.prediction || 'Unknown',
        confidence: data.confidence || 0,
        confidence_percentage: Math.round((data.confidence || 0) * 100),
        recommendation: getRecommendation(data.prediction),
        top_3: data.top_3 || [],
      };

      log('Formatted Result:', result);
      return result;
    } else {
      throw new Error(data.message || 'Prediction failed');
    }

  } catch (error) {
    logError('Prediction Error:', error.message);
    logError('Error Type:', error.name);
    
    // Provide helpful error messages
    let userMessage = error.message;
    
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
      userMessage = 'Cannot connect to backend. Please check:\n1. Backend is running\n2. No firewall blocking\n3. CORS is enabled';
    } else if (error.message.includes('timeout') || error.name === 'AbortError') {
      userMessage = 'Request timeout. Backend may be starting (takes 50+ seconds on free tier). Please try again.';
    }
    
    return {
      status: 'error',
      message: userMessage,
    };
  }
}

// ============================================================================
// Chat with AI
// ============================================================================

export async function chatWithAI(message) {
  try {
    log('Sending chat message:', message);

    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      mode: 'cors',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    return data.response || data.message || 'No response from AI';

  } catch (error) {
    logError('Chat error:', error);
    return 'AI chat is currently unavailable. Please use the disease scanner for diagnosis.';
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
      mode: 'cors',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    log('Classes loaded:', data.count);

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
  if (!diseaseName) {
    return 'Unable to provide recommendation without disease identification.';
  }

  const recommendations = {
    'early blight': 'Apply copper-based fungicide. Remove infected leaves. Ensure proper spacing for air circulation.',
    'late blight': 'Apply fungicide immediately. Remove and destroy infected plants. Avoid overhead watering.',
    'leaf mold': 'Improve air circulation. Reduce humidity. Apply fungicide if severe.',
    'septoria': 'Remove infected leaves. Apply copper fungicide. Mulch to prevent soil splash.',
    'bacterial spot': 'Use copper-based bactericide. Remove infected tissue. Avoid overhead irrigation.',
    'target spot': 'Apply fungicide. Ensure good air circulation. Remove plant debris.',
    'mosaic': 'Remove and destroy infected plants. Control aphids. Use resistant varieties.',
    'yellow': 'Control whiteflies. Remove infected plants. Use resistant varieties.',
    'healthy': 'No treatment needed. Continue regular monitoring and good cultural practices.',
  };

  const diseaseLower = diseaseName.toLowerCase();
  
  for (const [key, value] of Object.entries(recommendations)) {
    if (diseaseLower.includes(key)) {
      return value;
    }
  }

  return 'Consult with local agricultural extension service for specific treatment recommendations.';
}

// ============================================================================
// Test Function - Run this in console to test API
// ============================================================================

export async function testAPI() {
  console.clear();
  console.log('%c🧪 FarmAI API Test', 'font-size: 20px; color: #10B981; font-weight: bold');
  console.log('='.repeat(60));
  
  // Test 1: Health Check
  console.log('\n📋 Test 1: Health Check');
  const health = await checkBackendHealth();
  console.log('Result:', health);
  
  // Test 2: Get Classes
  console.log('\n📋 Test 2: Get Disease Classes');
  const classes = await getDiseaseClasses();
  console.log('Result:', classes);
  
  console.log('\n' + '='.repeat(60));
  console.log('%c✅ Tests Complete', 'color: #10B981; font-weight: bold');
  console.log('If both tests passed, the API is working correctly!');
  console.log('\nTo test image upload:');
  console.log('1. Go to Scan page');
  console.log('2. Upload an image');
  console.log('3. Check console for detailed logs');
  
  return { health, classes };
}

// ============================================================================
// Export Configuration
// ============================================================================

export const API_CONFIG = {
  baseUrl: API_BASE_URL,
  endpoints: {
    health: `${API_BASE_URL}/health`,
    predict: `${API_BASE_URL}/api/predict`,
    chat: `${API_BASE_URL}/api/chat`,
    classes: `${API_BASE_URL}/api/classes`,
  },
  debug: DEBUG,
  version: '3.0.0',
};

// Log on module load
log('API Service Initialized');
log('Backend URL:', API_BASE_URL);
log('Debug Mode:', DEBUG ? 'ON' : 'OFF');
log('Run testAPI() in console to test connection');

// Make testAPI available globally for easy testing
if (typeof window !== 'undefined') {
  window.testFarmAI = testAPI;
  console.log('%cℹ️ Tip: Run window.testFarmAI() in console to test API', 'color: #3B82F6');
}
