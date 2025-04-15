

import google.generativeai as genai
import pandas as pd
from config import GEMINI_API_KEY



def call_gemini_api(prompt, api_key=GEMINI_API_KEY, model_name="gemini-2.5-pro-exp-03-25"):
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
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Error calling Gemini API:", e)
        return None
    
    
