from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from api_training2.call_gemini import call_gemini_api, GEMINI_API_KEY
from ..services.analyze import analyze_data


router = APIRouter()






# FRONTEND PASSING
@router.post("/results/chat")
async def chat(request: Request, API_KEY: str = GEMINI_API_KEY):
    data = await request.json()
    data_summary = data.get("data_summary")
    message_history = data.get("message_history", [])

    # Build prompt from history
    prompt = f"""
    <b>Dataset Summary:</b><br>{data_summary}<br><br>
    <b>Conversation so far:</b><br>
    """

    for msg in message_history:
        role = "User" if msg["role"] == "user" else "AI"
        content = msg["content"]
        prompt += f"<b>{role}:</b> {content}<br>"

    # Call the Gemini API with full context
    try:
        AI_reply = call_gemini_api(prompt, api_key=API_KEY)
        return {"reply": AI_reply}
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"reply": "There was an error processing your message"}














# @router.post("/results/chat")
# async def chat(request : Request, API_KEY : str = GEMINI_API_KEY):
#     data = await request.json()
#     user_message = data.get("message")
#     data_summary = data.get("data_summary")


    
#     prompt = (
#         f"<b>Dataset Summary:</b><br>{data_summary}<br><br>"
#         f"<b>User:</b> {user_message}"
#     )



#     response = call_gemini_api(prompt, api_key = API_KEY)

#     try:
#         AI_reply = response

#         return{"reply" : AI_reply}
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return{"reply" : f"There was an error processing your message"}



