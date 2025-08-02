from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from api_training2.call_gemini import call_gemini_api, GEMINI_API_KEY
from ..services.analyze import analyze_data
import json
import logging      # The logging library is Python’s built-in module for tracking events that happen when software runs.


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/results/chat")
async def chat(request: Request, API_KEY: str = GEMINI_API_KEY):
    try:
        # Log the incoming request, ensures endpoint was hit
        logger.info("Received chat request")
        
        # Parse request data
        try:
            data = await request.json()
            logger.info(f"Request data keys: {list(data.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request body")
            # TODO: User-friendly error messages
        
        # Extract data with defaults
        data_summary = data.get("data_summary", "")     # default is an empty string
        message_history = data.get("message_history", [])   # default is an empty list
        sample_rows = data.get("sample_rows", [])       # # default is an empty string
        

        # Validate required data as a JSON response
        if not data_summary:
            return JSONResponse(
                status_code=400,
                content={"reply": "No data summary provided. Make sure you have uploaded a dataset."}
            )
        
        # Safely convert sample_rows to JSON string
        try:
            sample_rows_text = json.dumps(sample_rows[:5], indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to serialize sample_rows: {e}")
            sample_rows_text = "Error serializing sample data"
        
        logger.info(f"Sample rows preview: {sample_rows_text[:200]}...")

        # Build prompt from history
        prompt = f"""
        <b>Dataset Summary:</b><br>{data_summary}<br><br>
        <b>Sample Rows:</b><br><pre>{sample_rows_text}</pre><br>
        <b>Conversation so far:</b><br>
        """

        for msg in message_history:
            role = "User" if msg.get("role") == "user" else "AI"
            content = msg.get("content", "")
            prompt += f"<b>{role}:</b> {content}<br>"

        # Call the Gemini API with full context
        try:
            logger.info("Calling Gemini API...")
            AI_reply = call_gemini_api(prompt, api_key=API_KEY)
            logger.info("Gemini API call successful")
            return JSONResponse(content={"reply": AI_reply})
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return JSONResponse(
                status_code=500,
                content={"reply": f"AI service error: {str(e)}"}
            )
            
    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"reply": "An unexpected error occurred. Please try again."}
        )



