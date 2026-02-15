import warnings
import ast
warnings.simplefilter(action='ignore', category=FutureWarning)

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import datetime
import logging
import uuid
import os, io, sys
import pandas as pd
import numpy as np
import json
import requests


from app_quiksight.storage.redis import redis_client, save_chat_message, get_chat_history_openrouter
from api_training2.config import OPENROUTER_API_KEY, OPENROUTER_MODEL
from .upload import SYSTEM_INSTRUCTION, make_ai_context

logger = logging.getLogger("CHAT")


def log_section(title: str, char: str = "━"):
    line = char * 50
    logger.info(f"\n{line}\n  {title}\n{line}")

templates = Jinja2Templates(directory="app_quiksight/templates")
router = APIRouter()

# OpenRouter API endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DOWNLOAD_DIR = "data/downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)



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
                {thead}{tbody}
            </table>
        </div>
        <div class="text-xs text-gray-400 mt-2 text-center">
            Displaying top {len(df_preview)} of {len(df)} rows
        </div>
    </div>
    """




# ==============================================================================
#                           CODE EXECUTION
# ==============================================================================
def execute_user_code(code: str, df: pd.DataFrame):
    """Execute AI-generated code in a sandboxed environment."""
    logger.info("   Executing AI-generated code...")
    
    # ── Syntax pre-check ──────────────────────────────────────────────
    try:
        compile(code, "<ai_code>", "exec")
    except SyntaxError as se:
        logger.error(f"   ✗ Syntax error in AI code at line {se.lineno}: {se.msg}")
        logger.debug(f"   Full code:\n{code}")
        return {
            "response": {
                "execution_results": (
                    "<div class='alert alert-warning text-sm mt-2'>"
                    f"⚠️ The AI generated code with a syntax error (line {se.lineno}: {se.msg}). "
                    "Try rephrasing your question or asking for a simpler analysis."
                    "</div>"
                )
            }
        }

    # Define the display function available to the AI
    def display_table(data_frame, max_rows=10):
        if data_frame is None: return
        
        # Handle Series
        if isinstance(data_frame, pd.Series):
            data_frame = data_frame.reset_index()
            data_frame.columns = ['Index', 'Value'] if len(data_frame.columns) == 2 else data_frame.columns
        
        if isinstance(data_frame, pd.DataFrame):
            # Generate ID and save temp Parquet for download
            dl_id = str(uuid.uuid4())
            temp_path = f"{DOWNLOAD_DIR}/{dl_id}.parquet"
            data_frame.to_parquet(temp_path, engine="pyarrow")
            
            # Store path in Redis with TTL (1 hour)
            redis_client.set(f"download:{dl_id}", temp_path, ex=3600)
            
            # Print HTML with download link
            print(dataframe_to_styled_html(data_frame, download_id=dl_id, max_rows=max_rows))
        else:
            print(f"<p>{data_frame}</p>")

    local_env = {"df": df.copy(), "pd": pd, "np": np, "display_table": display_table}
    stdout_buffer = io.StringIO()
    sys.stdout = stdout_buffer

    try:
        # Whitelisted built-ins for safety
        safe_builtins = {
            "int": int, "float": float, "str": str, "bool": bool, "list": list, "dict": dict,
            "len": len, "range": range, "enumerate": enumerate, "zip": zip,
            "sorted": sorted, "sum": sum, "min": min, "max": max, "print": print, "type": type
        }
        
        exec(code, {"__builtins__": safe_builtins}, local_env)
        
        modified_df = local_env.get('df')
        if modified_df is not None and not modified_df.equals(df):
             return {"response": {"execution_results": stdout_buffer.getvalue()}, "modified_df": modified_df}

    except Exception as e:
        logger.error(f"   ✗ Code execution failed: {str(e)[:100]}")
        print(f"<div class='alert alert-warning text-sm mt-2'>Error running analysis: {str(e)[:100]}</div>")

    finally:
        sys.stdout = sys.__stdout__

    return {"response": {"execution_results": stdout_buffer.getvalue()}}




# ==============================================================================
#  ROUTES
# ==============================================================================

@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, sid: str):
    """sid in the chat_page function represents a query parameter in the URL (/chat?sid=) that FastAPI fills in automatically"""
    logger.info(f"\n     Chat page requested | Session: {sid[:8]}...")
    
    """We create redis_key again to check that the session exists
        We already created redis session from upload.py when a file was submitted
        This new one does another check anytime the chat page loads, so immediately after upload(redirect) and whenever page is refreshed
    """
    redis_key = f"session:{sid}"
    if not redis_client.exists(redis_key):
        logger.warning(f"    Session not found, redirecting home")
        return RedirectResponse(url="/", status_code=303)
    
    # 2. Retrieve Metadata
    """If the key redis_key exists, it gets everything that was hashed, this is now our new session data,
    the check and validation happens really fast, that's the nice thing with redis"""
    session_data = redis_client.hgetall(redis_key)
    

    try:
        columns = json.loads(session_data.get("columns", "[]"))
        preview_rows = json.loads(session_data.get("preview_rows", "[]"))
    except json.JSONDecodeError:
        columns, preview_rows = [], []
    
    """setting everything we call from redis session data to be rendered up on chat.html,
    so even when it is refreshed, the chat_page does its checks again and displays accordingly to if the sessionID exists in redis database"""
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "session_id": sid,
        "file_name": session_data.get("file_name", "Unknown"),
        "file_extension": session_data.get("file_extension", ""),
        "file_size": session_data.get("file_size", "0 KB"),
        "upload_date": session_data.get("upload_date", ""),
        "upload_time": session_data.get("upload_time", ""),
        "num_rows": session_data.get("num_rows"),
        "num_columns": len(columns),
        "columns": columns,
        "preview_rows": preview_rows,
        "cache_buster": datetime.datetime.now().timestamp()
    })




class ChatRequest(BaseModel):
    message: str



@router.post("/chat")
async def chat_endpoint(req: ChatRequest, sid: str):
    log_section("💬 CHAT MESSAGE RECEIVED", "─")
    
    redis_key = f"session:{sid}"
    if not redis_client.exists(redis_key):
        raise HTTPException(status_code=404, 
        detail="Conversation Not Found. Session may be expired or deleted")

    # 1. Load Data
    session_data = redis_client.hgetall(redis_key)
    df_path = session_data.get("dataframe_path")
    
    """Checks if the data actually still exists in the data path"""
    if not df_path or not os.path.exists(df_path):
        raise HTTPException(status_code=500, detail="Data file missing")
    
    """Reads dataframe each time so AI makes changes on the new updated dataframe instead of the old one"""
    try:
        df = pd.read_parquet(df_path)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to load data: {e}")

    """Gets data context every time a message is sent"""
    ai_context = session_data.get("ai_context")
    if not ai_context:
        """Only rebuilds if it is missing"""
        ai_context = make_ai_context(df, session_data.get("file_name", "Data"))

    """Loads chat history if theres any. Happens before new messages are sent"""
    past_history = get_chat_history_openrouter(sid, limit=10)

    """Build messages for OpenRouter API"""
    # System message with instructions and data context
    system_content = f"""{SYSTEM_INSTRUCTION}

### Context of the User's Data
{ai_context}

Fill in the three fields naturally: text_explanation, code_generated, should_execute."""

    messages = [
        {"role": "system", "content": system_content}
    ]
    
    # Add past conversation history
    messages.extend(past_history)
    
    # Add current user message
    messages.append({"role": "user", "content": req.message})

    """Send request to OpenRouter API"""
    try:
        logger.info(f"   Sending request to OpenRouter ({OPENROUTER_MODEL})...")
        
        response = requests.post(
            url=OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://127.0.0.1:8000",  # Optional: for OpenRouter analytics
                "X-Title": "Quiksight"  # Optional: for OpenRouter analytics
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": messages,
                "temperature": 0.0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "analysis_response",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "text_explanation": {
                                    "type": "string",
                                    "description": "HTML-formatted explanation text shown before code runs"
                                },
                                "code_generated": {
                                    "type": "string",
                                    "description": "Python code to execute. Use df, pd, np, display_table(). Keep under 40 lines."
                                },
                                "should_execute": {
                                    "type": "boolean",
                                    "description": "Whether the code_generated should be executed"
                                }
                            },
                            "required": ["text_explanation", "code_generated", "should_execute"],
                            "additionalProperties": False
                        }
                    }
                },
            },
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        """Finally saves user message after it has been sent"""
        save_chat_message(sid, "user", req.message)
        # ================================= USER MESSAGE SENT ENDS HERE ==================================

        # =================================== AI RESPONSE STARTS HERE =====================================
        # Parse AI Response
        try:
            ai_content = result['choices'][0]['message']['content']
            
            # Try 1: Clean JSON (models that support structured outputs)
            try:
                parsed = json.loads(ai_content)
                response_data = parsed[0] if isinstance(parsed, list) else parsed
                logger.info("   ✓ Parsed clean JSON response")
            except json.JSONDecodeError:
                # Try 2: Strip markdown fences (model wrapped JSON in ```json ... ```)
                try:
                    logger.warning("   ⚠ Direct JSON parse failed, trying markdown cleanup")
                    text = ai_content.replace('```json', '').replace('```', '').strip()
                    parsed = json.loads(text)
                    response_data = parsed[0] if isinstance(parsed, list) else parsed
                    logger.info("   ✓ Parsed JSON after markdown cleanup")
                except json.JSONDecodeError:
                    # Try 3: Model ignored schema entirely, returned plain text
                    # Wrap the raw text into the expected structure
                    logger.warning("   ⚠ Model returned plain text (no JSON). Wrapping as text_explanation.")
                    response_data = {
                        "text_explanation": ai_content,
                        "code_generated": "",
                        "should_execute": False
                    }
                
        except (IndexError, KeyError) as e:
            logger.error(f"   ✗ Failed to extract AI response from result: {e}")
            logger.error(f"   Raw result: {str(result)[:500]}")
            response_data = {
                 "text_explanation": "I processed your request but had trouble formatting the response.",
                 "code_generated": "",
                 "should_execute": False
             }
        
        ai_text = response_data.get('text_explanation', '')
        code = response_data.get('code_generated', '')
        should_execute = response_data.get('should_execute', False)
        
        # Fix double-escaped code from the AI model
        # Some models return code where newlines are literal \n text instead of real newlines
        if code:
            logger.info(f"   Code first 100 chars (repr): {repr(code[:100])}")
            # If code contains literal backslash-n (not real newlines), unescape them
            if '\\n' in code and '\n' not in code:
                code = (code
                    .replace('\\n', '\n')
                    .replace('\\t', '\t')
                    .replace("\\'", "'")
                    .replace('\\"', '"')
                )
                logger.info(f"   Fixed double-escaped code")
        
        # Save AI Message (using "model" role to maintain compatibility with existing history)
        save_chat_message(sid, "model", ai_text)

        execution_results = ""
        if should_execute and code:
            exec_result = execute_user_code(code, df)
            execution_results = exec_result["response"].get("execution_results", "")
            
            # Save updated DF if modified
            if "modified_df" in exec_result:
                exec_result["modified_df"].to_parquet(df_path)

        return {
            "response": {
                "text": ai_text,
                "code": code,
                "execution_results": execution_results
            }
        }

    except requests.exceptions.Timeout:
        logger.error("   ✗ OpenRouter request timed out")
        raise HTTPException(status_code=504, detail="AI request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        logger.error(f"   ✗ OpenRouter request failed: {e}")
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")
    except Exception as e:
        logger.error(f"   ✗ Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    












@router.get("/chat/download/{download_id}")
async def download_result(download_id: str):
    logger.info(f"\n     Download requested | ID: {download_id[:8]}...")
    
    # Fetch temp path from Redis
    temp_path = redis_client.get(f"download:{download_id}")
    
    if temp_path is None:
        logger.warning(f"   ✗ Download not found")
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Load temp DF from Parquet
        found_df = pd.read_parquet(temp_path)
        logger.info(f"   ✓ Found DF: {len(found_df)} rows")
    except Exception as e:
        logger.error(f"   ✗ Failed to load: {e}")
        raise HTTPException(status_code=500, detail="Failed to load download file")

    # Generate CSV stream
    logger.info(f"   Generating CSV...")
    stream = io.StringIO()
    found_df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=quiksight_export.csv"
    
    # Optional: Cleanup file after download (Redis TTL will handle key expiry)
    try:
        os.remove(temp_path)
        redis_client.delete(f"download:{download_id}")
    except Exception:
        pass  # If delete fails, TTL will still expire it
    
    logger.info("   ✓ Download ready")
    return response