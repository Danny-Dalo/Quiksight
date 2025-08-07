from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from api_training2.call_gemini import call_gemini_api, GEMINI_API_KEY
import json
import logging      # The logging library is Python’s built-in module for tracking events that happen when software runs.
import pandas as pd
# =================call function to get uploaded dataframe
from app_quiksight.routes.upload import get_uploaded_dataframe
 


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()




def parse_ai_response(response: str):
    """
    Parses AI response in the format:
    CHAT_RESPONSE: ...
    NEEDS_CODE: YES/NO
    CODE_TASK: ... (what the code should accomplish)
    """
    lines = response.strip().split('\n')
    chat_response, needs_code, code_task = '', 'NO', ''
    
    for line in lines:
        line = line.strip()  # Remove extra whitespace
        if line.startswith("CHAT_RESPONSE:"):
            # Extract everything after "CHAT_RESPONSE:" and remove leading/trailing spaces
            chat_response = line[len("CHAT_RESPONSE:"):].strip()
        elif line.startswith("NEEDS_CODE:"):
            # Extract YES/NO value
            needs_code = line[len("NEEDS_CODE:"):].strip()
        elif line.startswith("CODE_TASK:"):
            # Extract what the code should accomplish
            code_task = line[len("CODE_TASK:"):].strip()
    
    # Convert string to boolean for easier handling
    return chat_response, needs_code.upper() == "YES", code_task





@router.post("/results/chat")
async def chat(request: Request, API_KEY: str = GEMINI_API_KEY):
    try:
        logger.info("Received chat request")
        
        # Parse request data
        try:
            data = await request.json()
            logger.info(f"Request data keys: {list(data.keys())}")
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request body")
        
        # Extract data from post request and provide defaults for them
        data_summary = data.get("data_summary", "")
        message_history = data.get("message_history", [])
        sample_rows = data.get("sample_rows", [])
        
       
        if not data_summary:
            return JSONResponse(
                status_code=400,
                content={"reply": "No data summary provided. Make sure you have uploaded a dataset."}
            )
        
        # converting sample rows to JSON string
        try:
            sample_rows_text = json.dumps(sample_rows[:5], indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to serialize sample_rows: {e}")
            sample_rows_text = "Error serializing sample data"
        
        # Get the user's last message
        user_message = message_history[-1]["content"] if message_history else ""
        logger.info(f"USER LAST MESSAGE: {user_message}")
        
        
        conversation_context = ""
        if len(message_history) > 1:
            # Include last few messages for context (limit to avoid token overflow)
            recent_messages = message_history[-3:-1]  # Except the current message
            for msg in recent_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                conversation_context += f"{role.title()}: {content}\n"
        
        
        prompt = f"""
            You are a data analysis assistant. Analyze the user's request and respond in this EXACT format:

            CHAT_RESPONSE: [Write your helpful response to the user here]
            NEEDS_CODE: [Write exactly "YES" or "NO"]
            CODE_TASK: [If NEEDS_CODE is YES, write exactly what the Python code should accomplish]

            IMPORTANT FORMATTING RULES:
            - Each label must be on its own line
            - Use exactly these labels: CHAT_RESPONSE, NEEDS_CODE, CODE_TASK
            - For NEEDS_CODE, use only "YES" or "NO" (not "yes", "no", "true", "false")
            - If NEEDS_CODE is NO, you can leave CODE_TASK empty

            Guidelines for when to use NEEDS_CODE: YES:
            - Data filtering, sorting, grouping operations
            - Statistical calculations, aggregations
            - Data transformations or cleaning
            - Creating visualizations or charts
            - Mathematical computations on the data

            Use NEEDS_CODE: NO for:
            - General questions about the data
            - Explanations or interpretations
            - Questions about methodology
            - Requests for advice or recommendations

            Dataset Summary:
            {data_summary}

            Sample Data (first 5 rows):
            {sample_rows_text}

            Recent Conversation:
            {conversation_context}

            Current User Message: {user_message}

            Remember: Be specific in CODE_TASK about what operation should be performed on the dataframe 'df'.
            """

        # Single API call to get the structured response
        logger.info("Making initial API call to determine response and code needs")
        ai_response = call_gemini_api(prompt, api_key=API_KEY)
        
        # Parse the response
        chat_reply, needs_code, code_task = parse_ai_response(ai_response)
        
        # =====================================================
        code_snippet = None
        result_preview = None
        # =====================================================
        if needs_code and code_task:
            logger.info(f"Code needed. Task: {code_task}")
            
            # Generate ONLY pandas code based on the task
            code_prompt = f"""
                Generate clean, safe, and efficient pandas code for this task:

                TASK: {code_task}
                USER REQUEST: {user_message}

                Requirements:
                - Input dataframe is named 'df'
                - Store final result in variable 'result' 
                - Use proper pandas methods
                - Include error handling where appropriate

                Dataset context:
                {data_summary}

                Return ONLY the Python code in a markdown code block, no explanations, no comments.
            """
            

            logger.info("Generating code snippet")
            # API  call just for generating code
            code_snippet = call_gemini_api(code_prompt, api_key=API_KEY)
            print("==========GENERATED CODE==========")
            print(code_snippet)
            
            # Clean markdown formatting
            if code_snippet.startswith("```python"):
                code_snippet = code_snippet.replace("```python", "").replace("```", "").strip()


           
        
        
        response_data = {
            "reply": chat_reply,
            # "code": code_snippet,
            # "needs_code": needs_code,
            # "code_task": code_task if needs_code else None,
            # ==================================================================
            "code_result": result_preview if needs_code else None
            # ==================================================================
        }
        
        logger.info(f"Response prepared: NEEDS_CODE={needs_code}, HAS_CODE={code_snippet is not None}")
        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        # Return error response with proper structure
        return JSONResponse(
            status_code=500,
            content={
                "reply": "Sorry, I encountered an error processing your request. Please try again.",
                "code": None,
                "needs_code": False,
                "code_task": None
            }
        )
