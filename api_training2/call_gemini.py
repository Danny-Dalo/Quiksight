
import google.generativeai as genai
import pandas as pd
from .config import GEMINI_API_KEY



# def call_gemini_api(prompt, api_key, model_name="gemini-2.0-flash-thinking-exp-01-21"):
def call_gemini_api(prompt, api_key, model_name="gemini-2.5-flash-preview-05-20"):
# def call_gemini_api(prompt, api_key, model_name="gemini-2.5-pro-exp-03-25"):
    """
    Calls the Gemini API using the Google Generative AI Python SDK.

    Args:
        prompt (str): The prompt or instruction to send.
        api_key (str): Your Gemini API key.
        model_name (str): The model to use (default is gemini-2.5-pro-exp-03-25).

    Returns:
        str: The generated response text.
    """
    try:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing or not loaded.")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        generation_config = {
            "temperature": 0.0,
            # You can add other config parameters here if needed, e.g.:
            # "top_p": 0.95,
            # "top_k": 40,
            # "max_output_tokens": 2048, # Example limit
        }



        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text
    except Exception as e:
        print("Error calling Gemini API:", e)
        return None
    
    
