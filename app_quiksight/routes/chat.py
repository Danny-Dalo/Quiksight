

from fastapi import APIRouter, HTTPException, Request
import traceback
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
import os
from google import genai
from google.genai import types
from .upload import session_store   # import to use the generated session ID from upload
from api_training2.config import GEMINI_API_KEY

templates = Jinja2Templates(directory="app_quiksight/templates")



if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")

router = APIRouter()






@router.get("/chat", response_class = HTMLResponse)
async def chat_page(request : Request, sid : str):


    
    if sid not in session_store:    # at this point, session store has now been populated
        return RedirectResponse(url="/", status_code=303)

    
    
    session_data = session_store[sid]
    dataframe = session_data["df"]
    filename = session_data["file_name"]
    filesize = session_data["file_size"]
    file_extension = os.path.splitext(filename)[1]

    upload_date = session_data["upload_date"]
    upload_time = session_data["upload_time"]
    columns = session_data["columns"]


    


    return templates.TemplateResponse("chat.html", {
        "request" : request,
        "session_id" : sid,

        # File information
        "file_name": filename,
        "file_size" : filesize,
        "file_extension": file_extension,
        "upload_date": upload_date,
        "upload_time": upload_time,
        "num_rows" : len(dataframe),
        "num_columns" : len(dataframe.columns),
        "columns" : columns,

        "preview_rows" : session_data["preview_rows"]
    })



import pandas as pd
import numpy as np
import sys, io
from pandas.io.formats.style import Styler







import contextlib

def execute_user_code(code: str, df: pd.DataFrame):
    """Executes AI-generated code safely in a limited environment and returns its output."""

    # Prepare isolated execution environment
    local_env = {
        "df": df,
        "pd": pd,
        "np": np,
        # add any utility functions here
    }

    # Capture stdout and stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(code, {}, local_env)

        output = stdout_buffer.getvalue().strip()
        errors = stderr_buffer.getvalue().strip()

        return {
            "success": True,
            "output": output if output else "(No output produced)",
            "error": errors if errors else None
        }

    except Exception as e:
        return {
            "success": False,
            "output": None,
            "error": str(e)
        }







class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
async def chat_endpoint(req: ChatRequest, sid : str):
    if sid not in session_store:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    try:
        session_data = session_store[sid]
        chat_session = session_data["chat_session"]

        # Send user message to Gemini
        response = chat_session.send_message(req.message)

        results = {
            "text": "",
            "code": [],
            "execution_results": []
        }
        
        print(response.candidates[0].content.parts)
        # Parse parts: text, code, execution results
        for part in response.candidates[0].content.parts:
            # Capture plain text explanations (if any)
            if part.text:
                results["text"] += part.text



            # Capture structured code via function call
            if part.function_call and part.function_call.name == "return_code":
                code = part.function_call.args.get("code")
                if code:
                    results["code"].append(code)

                    exec_result = execute_user_code(code, session_data["df"])
                    results["execution_results"].append(exec_result["output"])




        return results

    except Exception as e:
        print("ERROR in chat endpoint:", e)
        print(traceback.format_exc())  # This will print the full error stack
        raise HTTPException(status_code=500, detail=str(e))




