"""
FarmAI Analytics Platform - AI Chatbot Agent
Google Gemini Pro integration for agricultural assistance
"""

import google.generativeai as genai
import time
import logging
from pathlib import Path

from typing import Tuple, Optional
from datetime import datetime

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"  
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "app.log"


# Setup logging
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FarmAIChatbot:
    """
    AI-powered chatbot for farmer assistance using Google Gemini
    """
    
    def __init__(self, api_key: str):
        """Initialize chatbot with API key"""
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required")
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.conversation_history = []
            logger.info("✅ Chatbot initialized successfully")
        except Exception as e:
            logger.error(f"❌ Chatbot initialization failed: {str(e)}")
            raise
    
    def generate_response(self, 
                         farmer_query: str, 
                         crop_type: str,
                         disease_name: Optional[str] = None,
                         language: str = 'English') -> Tuple[str, float]:
        """
        Generate AI response for farmer query
        
        Args:
            farmer_query: The farmer's question
            crop_type: Type of crop they're growing
            disease_name: Optional detected disease name for context
            language: Response language (English, Hindi, etc.)
        
        Returns:
            Tuple of (response_text, response_time)
        """
        start_time = time.time()
        
        # Build context-aware system prompt
        system_prompt = self._build_system_prompt(
            crop_type, disease_name, language
        )
        
        # Combine system prompt with user query
        full_prompt = f"{system_prompt}\n\nFarmer's Question: {farmer_query}"
        
        try:
            # Generate response
            response = self.model.generate_content(full_prompt)
            response_time = time.time() - start_time
            
            response_text = response.text
            
            # Add to conversation history
            self.conversation_history.append({
                'query': farmer_query,
                'response': response_text,
                'crop': crop_type,
                'disease': disease_name,
                'language': language,
                'time': response_time,
                'timestamp': datetime.now()
            })
            
            logger.info(f"✅ Response generated in {response_time:.2f}s")
            return response_text, response_time
        
        except Exception as e:
            logger.error(f"❌ Error generating response: {str(e)}")
            fallback_response = self._get_fallback_response(language)
            return fallback_response, 0.0
    
    def _build_system_prompt(self, 
                            crop_type: str, 
                            disease_name: Optional[str],
                            language: str) -> str:
        """Build context-aware system prompt"""
        
        base_prompt = f"""You are an expert agricultural advisor helping Indian farmers.

Current Context:
- Farmer is growing: {crop_type}
- Response language: {language}
"""
        
        if disease_name:
            base_prompt += f"- Detected disease: {disease_name}\n"
        
        base_prompt += """
Your Role & Guidelines:
1. Provide practical, actionable advice that farmers can implement immediately
2. Use simple, easy-to-understand language (avoid technical jargon)
3. Consider Indian farming practices, climate, and local conditions
4. Suggest cost-effective solutions suitable for small farmers
5. Include specific step-by-step instructions
6. Mention common Indian pesticide/fertilizer brands when relevant
7. Be empathetic and encouraging - farming is challenging work
8. If discussing chemicals, always mention safety precautions

Response Structure:
- Start with a brief, direct answer
- Provide detailed explanation
- Include actionable steps
- End with preventive measures or additional tips

Keep responses concise but complete (300-500 words).
"""
        
        return base_prompt
    
    def get_disease_treatment_plan(self, disease_name: str, crop_type: str) -> str:
        """
        Get detailed treatment plan for specific disease
        
        Args:
            disease_name: Name of the disease
            crop_type: Type of crop affected
        
        Returns:
            Detailed treatment plan
        """
        prompt = f"""
Provide a comprehensive treatment plan for {disease_name} affecting {crop_type} crops in India.

Include the following sections:
1. **Immediate Actions** (First 3 days)
   - What to do right away
   - Isolation and damage control

2. **Treatment Steps** (Week 1-3)
   - Specific pesticides/fungicides (Indian brands)
   - Application frequency and method
   - Dosage recommendations

3. **Prevention Measures**
   - How to prevent recurrence
   - Best practices for future

4. **Cost Estimation**
   - Approximate treatment cost per acre
   - Product costs in Indian Rupees

5. **Expected Recovery Time**
   - Timeline for improvement
   - When to see results

Format as bullet points for easy reading by farmers.
"""
        
        try:
            response = self.model.generate_content(prompt)
            logger.info(f"✅ Treatment plan generated for {disease_name}")
            return response.text
        except Exception as e:
            logger.error(f"❌ Error generating treatment: {str(e)}")
            return self._get_fallback_treatment()
    
    def get_crop_advice(self, crop_name: str, question: str, season: str = None) -> str:
        """
        Get general crop-specific advice
        
        Args:
            crop_name: Name of the crop
            question: Farmer's specific question
            season: Current growing season (optional)
        
        Returns:
            Crop-specific advice
        """
        prompt = f"""
Farmer in India is growing {crop_name}. They ask: {question}
"""
        
        if season:
            prompt += f"\nCurrent season: {season}"
        
        prompt += """

Provide practical advice covering:
1. Direct answer to their question
2. Seasonal considerations for Indian conditions
3. Common problems and solutions for this crop
4. Best practices specific to Indian farming
5. Tips for maximizing yield

Keep it practical and actionable.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"❌ Error generating crop advice: {str(e)}")
            return "Unable to generate advice. Please try again or consult a local agricultural expert."
    
    def get_multilingual_response(self, query: str, target_language: str) -> str:
        """
        Get response in specified language
        
        Args:
            query: The question to answer
            target_language: Target language (Hindi, Marathi, etc.)
        
        Returns:
            Response in target language
        """
        prompt = f"""
Answer this farmer's question in {target_language} language:

Question: {query}

Provide a practical, helpful answer in simple {target_language} that farmers can understand.
If technical terms are needed, explain them in simple words.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"❌ Error in multilingual response: {str(e)}")
            return f"[Error generating {target_language} response]"
    
    def _get_fallback_response(self, language: str = 'English') -> str:
        """Get fallback response when API fails"""
        
        fallbacks = {
            'English': """I apologize, but I'm experiencing technical difficulties right now. 
            
Please try again in a moment, or consider these general tips:
- Ensure proper watering and drainage
- Monitor your crops daily for early disease signs
- Consult your local agricultural extension office
- Use certified seeds and follow crop rotation

For urgent issues, please contact your local agricultural expert.""",
            
            'Hindi': """मुझे खेद है, लेकिन मुझे अभी तकनीकी समस्याएं हो रही हैं।

कृपया कुछ समय बाद पुनः प्रयास करें, या इन सामान्य सुझावों पर विचार करें:
- उचित पानी और जल निकासी सुनिश्चित करें
- रोग के शुरुआती संकेतों के लिए अपनी फसलों की दैनिक निगरानी करें
- अपने स्थानीय कृषि विस्तार कार्यालय से परामर्श करें"""
        }
        
        return fallbacks.get(language, fallbacks['English'])
    
    def _get_fallback_treatment(self) -> str:
        """Get fallback treatment recommendation"""
        return """
**General Disease Management Steps:**

1. **Immediate Actions:**
   - Remove and destroy infected plant parts
   - Isolate affected plants if possible
   - Improve air circulation around plants

2. **Treatment:**
   - Apply appropriate fungicide or pesticide
   - Follow product instructions carefully
   - Repeat application as recommended

3. **Prevention:**
   - Use disease-resistant varieties
   - Practice crop rotation
   - Maintain proper plant spacing
   - Ensure good field hygiene

4. **Consultation:**
   - Contact local agricultural extension officer
   - Visit nearest agricultural university
   - Consult with experienced farmers in your area

**Note:** Specific treatment depends on exact disease. Please consult an expert for precise diagnosis.
"""
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_history(self, limit: int = 10):
        """
        Get recent conversation history
        
        Args:
            limit: Number of recent conversations to return
        
        Returns:
            List of recent conversations
        """
        return self.conversation_history[-limit:]
    
    def get_conversation_summary(self) -> dict:
        """Get summary statistics of conversations"""
        if not self.conversation_history:
            return {
                'total_conversations': 0,
                'avg_response_time': 0,
                'most_discussed_crop': 'N/A'
            }
        
        total = len(self.conversation_history)
        avg_time = sum(c['time'] for c in self.conversation_history) / total
        
        # Find most discussed crop
        crops = [c['crop'] for c in self.conversation_history]
        most_common_crop = max(set(crops), key=crops.count) if crops else 'N/A'
        
        return {
            'total_conversations': total,
            'avg_response_time': round(avg_time, 2),
            'most_discussed_crop': most_common_crop,
            'languages_used': list(set(c['language'] for c in self.conversation_history))
        }


# Example usage and testing
if __name__ == "__main__":
    # This section is for testing only
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if api_key:
        chatbot = FarmAIChatbot(api_key)
        
        # Test query
        response, time_taken = chatbot.generate_response(
            farmer_query="My tomato plants have yellow leaves. What should I do?",
            crop_type="Tomato",
            language="English"
        )
        
        print(f"Response ({time_taken:.2f}s):")
        print(response)
        
        # Get summary
        summary = chatbot.get_conversation_summary()
        print(f"\nConversation Summary: {summary}")
    else:
        print("⚠️ GOOGLE_API_KEY not found in environment")
