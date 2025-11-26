## chatbot_agent.py

"""
FarmAI Analytics Platform - AI Chatbot Agent
Google Gemini integration for agricultural assistance

This module provides a safe initialization wrapper around the Google
Generative AI SDK. If the SDK or API key is missing, the agent runs in
fallback mode and returns helpful generic advice so the rest of the app
can continue to function (UI tests, demos, analytics pipeline, etc.).
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime
import time
import logging
import os

# Attempt to import google generative ai; allow fallback if unavailable
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Ensure logs directory
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "app.log"

# Configure logging
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FarmAIChatbot:
    """
    AI-powered chatbot wrapper.

    Behavior:
    - If Google Generative AI SDK and a valid API key are present, use the
      SDK to generate responses.
    - Otherwise, run in fallback mode and return safe, actionable templates.

    This makes the backend robust for local development and CI/testing.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        """Initialize the chatbot agent safely.

        Args:
            api_key: Google API key. If None, the agent will enter fallback mode.
            model_name: Model id to use (default: 'gemini-pro').
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.conversation_history = []
        self.available = False
        self._model_client = None

        # Initialize SDK only if available and api_key present
        if genai is None:
            logger.warning("Google generative AI SDK not installed; chatbot running in fallback mode")
            self.available = False
            return

        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not provided; chatbot running in fallback mode")
            self.available = False
            return

        try:
            # Configure SDK
            genai.configure(api_key=self.api_key)

            # create or reference a model client object if SDK exposes one
            # Some SDK versions use a direct function call; we wrap generic behavior
            try:
                self._model_client = genai.GenerativeModel(self.model_name)
            except Exception:
                # If SDK version does not provide GenerativeModel class, keep None
                self._model_client = None

            self.available = True
            logger.info("Chatbot initialized with model: %s", self.model_name)
        except Exception as e:
            logger.error("Failed to configure Google generative SDK: %s", str(e))
            self.available = False

    def generate_response(self,
                          farmer_query: str,
                          crop_type: str = "General",
                          disease_name: Optional[str] = None,
                          language: str = "English",
                          max_tokens: int = 512,
                          temperature: float = 0.7) -> Tuple[str, float]:
        """
        Generate a response. If the SDK is unavailable or fails, return a
        fallback response and response time 0.

        Returns:
            (response_text, response_time_seconds)
        """
        start = time.time()

        # Build prompt
        system_prompt = self._build_system_prompt(crop_type, disease_name, language)
        user_section = f"Farmer's Question: {farmer_query}\n"
        prompt = system_prompt + "\n" + user_section

        if not self.available:
            logger.info("SDK not available - returning fallback response")
            return self._get_fallback_response(language), 0.0

        # Try SDK generation with robust error handling
        try:
            # Prefer using model client if available
            if self._model_client is not None and hasattr(self._model_client, 'generate_content'):
                # Some SDKs accept a single string; others accept structured args.
                response = self._model_client.generate_content(prompt)
                resp_text = getattr(response, 'text', str(response))
            else:
                # Generic fallback: use genai.generate (older/newer SDKs vary)
                # Try common entrypoints sequentially
                resp_text = None
                if hasattr(genai, 'generate'):
                    response = genai.generate(prompt=prompt, model=self.model_name, temperature=temperature)
                    # response may be dict-like
                    resp_text = (response.get('output') if isinstance(response, dict) else str(response))
                elif hasattr(genai, 'create'):
                    response = genai.create(prompt=prompt, model=self.model_name)
                    resp_text = getattr(response, 'text', str(response))
                else:
                    # As last resort, attempt the GenerativeModel class constructor each time
                    try:
                        client = genai.GenerativeModel(self.model_name)
                        response = client.generate_content(prompt)
                        resp_text = getattr(response, 'text', str(response))
                    except Exception as e:
                        raise RuntimeError("No supported genai entrypoint found: %s" % str(e))

            response_time = time.time() - start

            # Sanitize response
            if not resp_text:
                raise RuntimeError("Empty response from model")

            # Save to history
            self.conversation_history.append({
                'query': farmer_query,
                'response': resp_text,
                'crop': crop_type,
                'disease': disease_name,
                'language': language,
                'time': response_time,
                'timestamp': datetime.now()
            })

            logger.info("Response generated in %.2fs", response_time)
            return resp_text, response_time

        except Exception as e:
            logger.error("Error generating response: %s", str(e))
            return self._get_fallback_response(language), 0.0

    def _build_system_prompt(self, crop_type: str, disease_name: Optional[str], language: str) -> str:
        base = (
            "You are an expert agricultural advisor helping Indian farmers.\n\n"
            f"Current Context:\n- Farmer is growing: {crop_type}\n- Response language: {language}\n"
        )
        if disease_name:
            base += f"- Detected disease: {disease_name}\n"

        base += (
            "\nYour Role & Guidelines:\n"
            "1. Provide practical, actionable advice that farmers can implement immediately\n"
            "2. Use simple, easy-to-understand language (avoid technical jargon)\n"
            "3. Consider Indian farming practices, climate, and local conditions\n"
            "4. Suggest cost-effective solutions suitable for small farmers\n"
            "5. Include specific step-by-step instructions\n"
            "6. Mention common Indian pesticide/fertilizer brands when relevant\n"
            "7. Be empathetic and encouraging\n"
            "8. If discussing chemicals, always mention safety precautions\n\n"
            "Response Structure:\n- Start with a brief direct answer\n- Provide detailed explanation\n- Include actionable steps\n- End with preventive measures or additional tips\n"
        )
        return base

    def get_disease_treatment_plan(self, disease_name: str, crop_type: str, language: str = "English") -> str:
        prompt = (
            f"Provide a comprehensive treatment plan for {disease_name} affecting {crop_type} crops in India.\n"
            "Include Immediate Actions, Treatment Steps, Prevention Measures, Cost Estimation and Expected Recovery Time. "
            "Format as bullet points that are easy to follow."
        )

        if not self.available:
            logger.info("Returning fallback treatment plan for %s", disease_name)
            return self._get_fallback_treatment()

        try:
            # Use same generation path as generate_response
            if self._model_client is not None and hasattr(self._model_client, 'generate_content'):
                response = self._model_client.generate_content(prompt)
                return getattr(response, 'text', str(response))
            elif hasattr(genai, 'generate'):
                response = genai.generate(prompt=prompt, model=self.model_name)
                return response.get('output') if isinstance(response, dict) else str(response)
            else:
                client = genai.GenerativeModel(self.model_name)
                response = client.generate_content(prompt)
                return getattr(response, 'text', str(response))
        except Exception as e:
            logger.error("Error generating treatment plan: %s", str(e))
            return self._get_fallback_treatment()

    def get_crop_advice(self, crop_name: str, question: str, season: Optional[str] = None, language: str = "English") -> str:
        prompt = f"Farmer in India is growing {crop_name}. They ask: {question}\n"
        if season:
            prompt += f"Current season: {season}\n"
        prompt += "Provide practical, actionable advice covering direct answer, seasonal considerations, common problems and solutions, and best practices."

        if not self.available:
            return "Unable to generate model-backed advice right now. Here is general guidance: \n- Monitor watering, ensure drainage. \n- Maintain spacing, use certified seeds. \n- Consult local extension office for region-specific products."

        try:
            if self._model_client is not None and hasattr(self._model_client, 'generate_content'):
                response = self._model_client.generate_content(prompt)
                return getattr(response, 'text', str(response))
            elif hasattr(genai, 'generate'):
                response = genai.generate(prompt=prompt, model=self.model_name)
                return response.get('output') if isinstance(response, dict) else str(response)
            else:
                client = genai.GenerativeModel(self.model_name)
                response = client.generate_content(prompt)
                return getattr(response, 'text', str(response))
        except Exception as e:
            logger.error("Error generating crop advice: %s", str(e))
            return "Unable to generate advice. Please try again later or consult a local agricultural expert."

    def get_multilingual_response(self, query: str, target_language: str) -> str:
        prompt = (
            f"Answer this farmer's question in {target_language} language:\n\nQuestion: {query}\n\n"
            "Provide a practical, helpful answer in simple language that farmers can understand."
        )

        if not self.available:
            # Very short fallback translation notice
            return f"[Model unavailable] Please seek local advice. (Requested language: {target_language})"

        try:
            if self._model_client is not None and hasattr(self._model_client, 'generate_content'):
                response = self._model_client.generate_content(prompt)
                return getattr(response, 'text', str(response))
            elif hasattr(genai, 'generate'):
                response = genai.generate(prompt=prompt, model=self.model_name)
                return response.get('output') if isinstance(response, dict) else str(response)
            else:
                client = genai.GenerativeModel(self.model_name)
                response = client.generate_content(prompt)
                return getattr(response, 'text', str(response))
        except Exception as e:
            logger.error("Error generating multilingual response: %s", str(e))
            return f"[Error generating {target_language} response]"

    def _get_fallback_response(self, language: str = "English") -> str:
        fallbacks: Dict[str, str] = {
            'English': (
                "I apologize, but I'm experiencing technical difficulties right now.\n\n"
                "Please try again in a moment, or consider these general tips:\n"
                "- Ensure proper watering and drainage\n"
                "- Monitor your crops daily for early disease signs\n"
                "- Consult your local agricultural extension office\n"
                "- Use certified seeds and follow crop rotation\n\n"
                "For urgent issues, contact your local agricultural expert."
            ),
            'Hindi': (
                "मुझे खेद है, लेकिन मुझे अभी तकनीकी समस्याएं हो रही हैं। कृपया कुछ समय बाद पुनः प्रयास करें।"
            )
        }
        return fallbacks.get(language, fallbacks['English'])

    def _get_fallback_treatment(self) -> str:
        return (
            "General Disease Management Steps:\n\n"
            "1. Remove infected plant parts and destroy them.\n"
            "2. Improve air circulation and avoid overhead watering.\n"
            "3. Apply appropriate fungicide or bactericide according to label instructions.\n"
            "4. Consult local agricultural extension officers for region-specific products.\n"
        )

    def clear_history(self) -> None:
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_history(self, limit: int = 10):
        return self.conversation_history[-limit:]

    def get_conversation_summary(self) -> Dict:
        if not self.conversation_history:
            return {'total_conversations': 0, 'avg_response_time': 0, 'most_discussed_crop': 'N/A'}
        total = len(self.conversation_history)
        avg_time = sum(c.get('time', 0) for c in self.conversation_history) / total
        crops = [c.get('crop') for c in self.conversation_history]
        most_common_crop = max(set(crops), key=crops.count) if crops else 'N/A'
        return {
            'total_conversations': total,
            'avg_response_time': round(avg_time, 2),
            'most_discussed_crop': most_common_crop,
            'languages_used': list({c.get('language') for c in self.conversation_history})
        }


# Test block (safe to run locally; will use fallback if SDK/API not configured)
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv('GOOGLE_API_KEY')
    bot = FarmAIChatbot(api_key=key)
    resp, t = bot.generate_response('My tomato leaves are yellow. What should I do?', crop_type='Tomato')
    print('Response time:', t)
    print(resp)
