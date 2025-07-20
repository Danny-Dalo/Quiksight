from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from api_training2.call_gemini import call_gemini_api, GEMINI_API_KEY


router = APIRouter()

@router.post("/results/chat")
async def chat(request : Request, API_KEY : str = GEMINI_API_KEY):
    data = await request.json()
    user_message = data.get("message")

    # AI tuning prompt
    system_instruction = (
    "You are an AI assistant whose responses will be rendered directly in an HTML page. "
    "Please use HTML tags for formatting (e.g., <b>, <i>, <ul>, <br>) instead of Markdown or plain text. "
    "Do not use Markdown formatting. Respond only with HTML-safe content."
    )
    prompt = f"{system_instruction}\n\nUser: {user_message}"

    response = call_gemini_api(prompt, api_key = API_KEY)

    try:
        AI_reply = response

        return{"reply" : AI_reply}
    except Exception as e:
        print(f"Error occurred: {e}")
        return{"reply" : f"There was an error processing your message"}

