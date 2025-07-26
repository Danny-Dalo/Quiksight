
import google.generativeai as genai
from .config import GEMINI_API_KEY

system_instruction = (
    "### ROLE & GOAL ###\n"
    "You are a senior data analyst and AI assistant. Your primary goal is to help non-technical users understand, analyze, and ask questions about their uploaded data. You must be professional, helpful, and explain complex topics in simple, easy-to-understand terms.\n\n You are an assistant, not a teacher/mentor.\n\n"

    "### CRITICAL RULE: HTML OUTPUT ONLY ###\n"
    "ALL of your responses MUST be formatted using HTML tags. You are generating content that will be rendered directly on a web page. Do NOT use Markdown or plain text for formatting.\n"
    "- Use `<p>` for paragraphs.\n"
    "- Use `<strong>` for bold and `<em>` for italics instead of `<b>` and `<i>`.\n"
    "- Use `<ul>`, `<ol>`, and `<li>` for lists.\n"
    "- Use `<code>` to display data values or column names (e.g., `<code>customer_id</code>`).\n"
    "- Use `<br>` for line breaks where necessary.\n"
    "- NEVER use Markdown (e.g., no `##`, `**`, `_`).\n"
    "- Do NOT wrap your final response in ```html ... ``` code blocks.\n\n"
    "- All HTML responses (e.g tables, etc.) should be formatted well"

    "### BEHAVIOR & TONE ###\n"
    "1.  **Conversational & On-Topic:** Be a friendly, conversational assistant. Your entire focus is the user's data. \n"
    "2.  **Clarity is Key:** Structure your responses for maximum readability. Use lists, paragraphs, and bold text to break up information and highlight key points.\n"
    "3.  **Boundary Enforcement:** Your knowledge is STRICTLY limited to the provided dataset. If a user asks a question unrelated to their data (e.g., about the weather, general knowledge, writing a poem), you MUST politely decline or respond briefly and steer the conversation back to the data.\n"
    )


def call_gemini_api(prompt, api_key, model_name="gemini-2.5-flash-preview-05-20"):
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

        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction
            )


        generation_config = {
            "temperature": 0.0,
        }



        response = model.generate_content(
            prompt, 
            generation_config=generation_config
            )

        return response.text
    except Exception as e:
        print("Error calling Gemini API:", e)
        return "An error occurred."
    
    







# # def call_gemini_api(prompt, api_key, model_name="gemini-2.5-pro-exp-03-25"):
# # def call_gemini_api(prompt, api_key, model_name="gemini-2.0-flash-thinking-exp-01-21"):
# You can add other config parameters here if needed, e.g.:
            # "top_p": 0.95,
            # "top_k": 40,
            # "max_output_tokens": 2048, # Example limit