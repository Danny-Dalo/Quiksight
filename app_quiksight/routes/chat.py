import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from fastapi import APIRouter, HTTPException, Request
import traceback
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
import datetime
import logging

import os, io, sys
import pandas as pd
import numpy as np
import json
import uuid


from .upload import session_store 
# from api_training2.config import GEMINI_API_KEY

# Configure logging (same format as upload.py)
logger = logging.getLogger("CHAT")


def log_section(title: str, char: str = "━"):
    """Log a visual section divider for better readability."""
    line = char * 50
    logger.info(f"\n{line}\n  {title}\n{line}")


templates = Jinja2Templates(directory="app_quiksight/templates")

router = APIRouter()

# --- STYLING & DOWNLOAD BUTTON ---
def dataframe_to_styled_html(df: pd.DataFrame, download_id: str = None, max_rows=10):
    """Convert a Pandas DataFrame into a styled HTML table with a download button."""
    # Limit rows for display
    df_preview = df.head(max_rows)

    thead = "<thead class='bg-gray-50 text-gray-600 font-medium'><tr>"
    for col in df_preview.columns:
        thead += f"<th>{col}</th>"
    thead += "</tr></thead>"

    tbody = "<tbody>"
    for _, row in df_preview.iterrows():
        tbody += "<tr>"
        for col in df_preview.columns:
            val = row[col]
            if pd.isna(val) or val == "":
                cell = '<span class="text-gray-400 italic">N/A</span>'
            elif len(str(val)) > 60:
                safe_val = str(val).replace('"', '&quot;')
                cell = f'<div class="tooltip tooltip-bottom text-left" data-tip="{safe_val}"><span class="truncate max-w-xs block">{safe_val}</span></div>'
            else:
                cell = str(val)
            tbody += f"<td>{cell}</td>"
        tbody += "</tr>"
    tbody += "</tbody>"

    # Download Button Logic
    download_html = ""
    if download_id:
        download_html = f"""
        <div class="flex justify-between items-center mb-2 px-1">
            <span class="text-xs text-gray-500 font-mono"></span>
            <a href="/chat/download/{download_id}" target="_blank" class="btn btn-sm btn-outline btn-accent gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download CSV
            </a>
        </div>
        """

    return f"""
    <div class="w-full max-w-7xl mx-auto bg-white rounded-xl shadow-sm border border-gray-200 p-3 mt-3">
        {download_html}
        <div class="overflow-x-auto">
            <table class="table table-zebra table-pin-rows table-xs sm:table-sm">
                {thead}
                {tbody}
            </table>
        </div>
        <div class="text-xs text-gray-400 mt-2 text-center">
            Displaying out of {len(df_preview)} of {len(df)} rows
        </div>
    </div>
    """

# --- EXECUTION ENGINE ---
def execute_user_code(code: str, session_data: dict):
    """Execute AI-generated code in a sandboxed environment."""
    logger.info("   Executing AI-generated code...")
    logger.debug(f"   Code length: {len(code)} chars")
    df = session_data["df"]
    
    # Helper function available to the AI
    def display_table(data_frame, max_rows=10):
        """Display a DataFrame as a styled HTML table with download button."""
        if data_frame is None:
            print("<p class='text-gray-500 italic'>No data to display.</p>")
            return
            
        # Convert Series to DataFrame
        if isinstance(data_frame, pd.Series):
            data_frame = data_frame.reset_index()
            data_frame.columns = ['Index', 'Value'] if len(data_frame.columns) == 2 else data_frame.columns
        
        if not isinstance(data_frame, pd.DataFrame):
            print(f"<p>{data_frame}</p>")
            return

        # Generate ID and Save for download
        dl_id = str(uuid.uuid4())
        if "downloads" not in session_data:
            session_data["downloads"] = {}
            
        session_data["downloads"][dl_id] = data_frame.copy()
        
        # Print HTML
        print(dataframe_to_styled_html(data_frame, download_id=dl_id, max_rows=max_rows))

    # Sandbox environment
    local_env = {
        "df": df.copy(), 
        "pd": pd, 
        "np": np, 
        "display_table": display_table
    }

    stdout_buffer = io.StringIO()
    sys.stdout = stdout_buffer

    try:
        # Extended whitelist of safe builtins for data operations
        safe_builtins = {
            # Types
            "int": int, "float": float, "str": str, "bool": bool,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            # Iteration
            "len": len, "range": range, "enumerate": enumerate, "zip": zip,
            "sorted": sorted, "reversed": reversed, "filter": filter, "map": map,
            # Math/aggregation
            "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
            "any": any, "all": all,
            # Utility
            "print": print, "type": type, "isinstance": isinstance,
            "Exception": Exception, "ValueError": ValueError, "KeyError": KeyError,
            # Formatting
            "format": format,
        }
        
        exec(code, {"__builtins__": safe_builtins}, local_env)
        logger.info("   ✓ Code executed successfully")
        
        modified_df = local_env.get('df')
        # Only return modified_df if it actually changed
        if modified_df is not None and not modified_df.equals(df):
             return {
                "response": {"execution_results": stdout_buffer.getvalue()},
                "modified_df": modified_df
            }

    except Exception as e:
        # Log full traceback to terminal for debugging
        logger.error(f"   ✗ Code execution failed: {type(e).__name__}: {str(e)[:100]}")
        print(traceback.format_exc(), file=sys.__stderr__)
        
        # Show user-friendly error with hint
        error_type = type(e).__name__
        error_msg = str(e)[:100]  # Truncate long errors
        print(f"""<div class='alert alert-warning text-sm mt-2'>
            <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-5 w-5" fill="none" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span>Something went wrong ({error_type}). Try rephrasing your question.</span>
        </div>""")

    finally:
        sys.stdout = sys.__stdout__

    return {
        "response": {
            "execution_results": stdout_buffer.getvalue(),
        }
    }

# --- ROUTES ---

@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, sid: str):
    logger.info(f"\n💬 Chat page requested | Session: {sid[:8]}...")
    
    if sid not in session_store:
        logger.warning(f"   ✗ Session not found, redirecting to home")
        return RedirectResponse(url="/", status_code=303)
    
    session_data = session_store[sid]
    logger.info(f"   ✓ Session found: {session_data['file_name']}")
    
    # Ensure downloads dict exists (backward compatibility)
    if "downloads" not in session_data:
        session_data["downloads"] = {}

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "session_id": sid,
        "file_name": session_data["file_name"],
        "file_size": session_data["file_size"],
        "file_extension": os.path.splitext(session_data["file_name"])[1],
        "upload_date": session_data["upload_date"],
        "upload_time": session_data["upload_time"],
        "num_rows": len(session_data["df"]),
        "num_columns": len(session_data["df"].columns),
        "columns": session_data["columns"],
        "preview_rows": session_data["preview_rows"],
        "cache_buster": datetime.datetime.now().timestamp()
    })





class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat_endpoint(req: ChatRequest, sid: str):
    log_section("💬 CHAT MESSAGE RECEIVED", "─")
    logger.info(f"   Session: {sid[:8]}...")
    logger.info(f"   Message: {req.message[:80]}{'...' if len(req.message) > 80 else ''}")
    
    if sid not in session_store:
        logger.error(f"   ✗ Session not found")
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        session_data = session_store[sid]
        chat_session = session_data["chat_session"]

        logger.info("   Sending to Gemini API...")
        response = chat_session.send_message(req.message)
        logger.info("   ✓ Response received from Gemini")
        
        # Clean JSON parsing
        response_text = response.text.replace('```json', '').replace('```', '')
        try:
            response_data = json.loads(response_text)
        except:
            # Fallback if AI output isn't perfect JSON
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            response_data = json.loads(response_text[start:end])

        ai_text = response_data[0]['text_explanation']
        code = response_data[0]['code_generated']
        should_execute = response_data[0]['should_execute']
        
        logger.info(f"   AI response parsed:")
        logger.info(f"      Text: {len(ai_text)} chars")
        logger.info(f"      Code: {'Yes' if code else 'No'} ({len(code)} chars)")
        logger.info(f"      Execute: {should_execute}")

        execution_results = ""
        if should_execute and code:
            # Pass session_data to access the downloads dict
            exec_result = execute_user_code(code, session_data)

            if "modified_df" in exec_result:
                session_data["df"] = exec_result["modified_df"]
                logger.info("   📊 DataFrame was modified by code execution")

            execution_results = exec_result["response"].get("execution_results", "")

        logger.info("   ✓ Response ready")
        return {
            "response": {
                "text": ai_text,
                "code": code,
                "execution_results": execution_results
            }
        }
    except Exception as e:
        logger.error(f"   ✗ Chat endpoint error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))






# DOWNLOADING GENERATED TABLES
@router.get("/chat/download/{download_id}")
async def download_result(download_id: str):
    logger.info(f"\n📥 Download requested | ID: {download_id[:8]}...")
    
    # Search all sessions for this download ID (simple lookup)
    # In a database app, you'd query by ID.
    found_df = None
    for sid, data in session_store.items():
        if "downloads" in data and download_id in data["downloads"]:
            found_df = data["downloads"][download_id]
            logger.info(f"   ✓ Found in session: {sid[:8]}...")
            break
    
    if found_df is None:
        logger.warning(f"   ✗ Download not found")
        raise HTTPException(status_code=404, detail="File not found")

    logger.info(f"   Generating CSV ({len(found_df)} rows)...")
    stream = io.StringIO()
    found_df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=quiksight_export.csv"
    logger.info("   ✓ Download ready")
    return response

