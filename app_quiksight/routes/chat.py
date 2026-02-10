import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import datetime
import logging
import uuid
import os, io, sys, textwrap
import pandas as pd
import numpy as np
import json
import requests

# Gemini SDK for fallback
from google import genai
from google.genai import types

from app_quiksight.storage.redis import redis_client, save_chat_message, get_chat_history_openrouter, get_chat_history
from api_training2.config import OPENROUTER_API_KEY, OPENROUTER_MODEL, GEMINI_API_KEY
from .upload import SYSTEM_INSTRUCTION, make_ai_context

logger = logging.getLogger("CHAT")


def log_section(title: str, char: str = "━"):
    line = char * 50
    logger.info(f"\n{line}\n  {title}\n{line}")

templates = Jinja2Templates(directory="app_quiksight/templates")
router = APIRouter()

# OpenRouter API endpoint (primary)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Gemini client for fallback
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

DOWNLOAD_DIR = "data/downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _to_json_safe(val):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if pd.isna(val):
        return None
    if hasattr(val, 'item'):
        return val.item()
    return val


def auto_detect_plotly(df):
    """Auto-detect the best Plotly chart from a DataFrame. Returns dict or None."""
    try:
        cols = df.columns.tolist()
        if len(cols) < 2 or len(df) == 0:
            return None

        # Convert Period columns to timestamps to avoid PeriodDtype errors
        for c in cols:
            if hasattr(df[c], 'dt') and isinstance(df[c].dtype, pd.PeriodDtype):
                df = df.copy()
                df[c] = df[c].dt.to_timestamp()

        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        date_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
        cat_cols = [c for c in cols if c not in numeric_cols and c not in date_cols]

        if not numeric_cols:
            return None

        palette = ['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6']
        traces = []
        tick_angle = -30

        if date_cols and numeric_cols:
            x_vals = df[date_cols[0]].astype(str).tolist()
            tick_angle = 0
            for i, y_col in enumerate(numeric_cols[:4]):
                traces.append({
                    "type": "scatter", "mode": "lines+markers",
                    "x": x_vals, "y": [_to_json_safe(v) for v in df[y_col]],
                    "name": y_col,
                    "line": {"color": palette[i % len(palette)], "width": 2.5},
                    "marker": {"size": 5}
                })
        elif cat_cols and numeric_cols:
            x_vals = df[cat_cols[0]].astype(str).tolist()
            for i, y_col in enumerate(numeric_cols[:4]):
                traces.append({
                    "type": "bar",
                    "x": x_vals, "y": [_to_json_safe(v) for v in df[y_col]],
                    "name": y_col,
                    "marker": {"color": palette[i % len(palette)]}
                })
        elif len(numeric_cols) >= 2:
            for i, y_col in enumerate(numeric_cols[1:4]):
                traces.append({
                    "type": "scatter", "mode": "markers",
                    "x": [_to_json_safe(v) for v in df[numeric_cols[0]]],
                    "y": [_to_json_safe(v) for v in df[y_col]],
                    "name": y_col,
                    "marker": {"color": palette[i % len(palette)], "size": 8, "opacity": 0.7}
                })

        if not traces:
            return None

        layout = {
            "margin": {"l": 55, "r": 20, "t": 25, "b": 70},
            "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": "Inter, sans-serif", "size": 12, "color": "#6b7280"},
            "xaxis": {"gridcolor": "#f3f4f6", "linecolor": "#e5e7eb", "zerolinecolor": "#e5e7eb", "tickangle": tick_angle},
            "yaxis": {"gridcolor": "#f3f4f6", "linecolor": "#e5e7eb", "zerolinecolor": "#e5e7eb"},
            "legend": {"orientation": "h", "y": -0.25, "x": 0.5, "xanchor": "center"},
            "hoverlabel": {"bgcolor": "white", "font": {"family": "Inter", "size": 13}},
            "bargap": 0.25,
            "hovermode": "x unified" if date_cols else "closest",
            "showlegend": len(traces) > 1
        }
        return {"data": traces, "layout": layout}

    except Exception as e:
        logger.warning(f"   ⚠ Chart auto-detect failed: {e}")
        return None


def dataframe_to_styled_html(df: pd.DataFrame, download_id: str = None, max_rows=10):
    """Convert a DataFrame into a tabbed HTML view with table and auto-generated Plotly chart."""
    df_preview = df.head(max_rows)
    viz_id = str(uuid.uuid4())[:8]

    # Build minimalistic table
    thead = "<thead><tr>"
    for col in df_preview.columns:
        thead += f"<th>{col}</th>"
    thead += "</tr></thead>"

    tbody = "<tbody>"
    for _, row in df_preview.iterrows():
        tbody += "<tr>"
        for col in df_preview.columns:
            val = row[col]
            if pd.isna(val) or val == "":
                cell = '<span class="qs-na">\u2014</span>'
            elif len(str(val)) > 60:
                safe_val = str(val).replace('"', '&quot;')
                cell = f'<div class="tooltip tooltip-bottom text-left" data-tip="{safe_val}"><span class="truncate max-w-xs block">{safe_val}</span></div>'
            else:
                cell = str(val)
            tbody += f"<td>{cell}</td>"
        tbody += "</tr>"
    tbody += "</tbody>"

    # Download button in tab bar
    dl_html = ""
    if download_id:
        dl_html = (
            f'<a href="/chat/download/{download_id}" target="_blank" class="qs-download-btn" title="Download CSV">'
            '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">'
            '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>'
            '</svg> CSV</a>'
        )

    # Auto-detect Plotly chart
    chart_data = auto_detect_plotly(df_preview)
    chart_tab = ""
    chart_panel = ""
    if chart_data:
        chart_json = json.dumps(chart_data).replace("</", "<\\/")
        chart_tab = (
            f'<button class="qs-tab" onclick="qsTabSwitch(this,\'chart-{viz_id}\')">'
            '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 17V13"/><path d="M12 17V9"/><path d="M17 17V6"/></svg> Chart</button>'
        )
        chart_panel = (
            f'<div class="qs-panel" id="chart-{viz_id}" style="display:none">'
            f'<div class="qs-plotly-chart" id="plotly-{viz_id}" style="width:100%;min-height:320px;"></div>'
            f'<script type="application/json" class="qs-plotly-data">{chart_json}</script></div>'
        )

    table_tab = (
        f'<button class="qs-tab qs-tab-active" onclick="qsTabSwitch(this,\'table-{viz_id}\')">'
        '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/></svg> Table</button>'
    )

    row_info = f'{len(df_preview)} of {len(df)} rows'

    return (
        f'<div class="qs-viz" id="viz-{viz_id}">'
        f'<div class="qs-tab-bar">{table_tab}{chart_tab}{dl_html}</div>'
        f'<div class="qs-panel qs-panel-active" id="table-{viz_id}">'
        f'<div style="overflow-x:auto"><table class="qs-table">{thead}{tbody}</table></div>'
        f'<div class="qs-row-info">{row_info}</div></div>'
        f'{chart_panel}</div>'
    )




# ==============================================================================
#                           CODE EXECUTION
# ==============================================================================
def preprocess_code(code: str) -> str:
    """Fix common AI-generated code issues before execution."""
    # Normalize indentation (fixes mixed tabs/spaces and unexpected indents)
    code = textwrap.dedent(code)
    # Remove leading/trailing blank lines
    lines = code.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def execute_user_code(code: str, df: pd.DataFrame):
    """Execute AI-generated code in a sandboxed environment.
    Returns dict with 'response', optionally 'modified_df', and 'error' if execution failed."""
    logger.info("   Executing AI-generated code...")
    
    # Pre-process code to fix common issues
    code = preprocess_code(code)
    
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
            "sorted": sorted, "sum": sum, "min": min, "max": max, "print": print, "type": type,
            "round": round, "abs": abs, "map": map, "filter": filter, "isinstance": isinstance,
            "set": set, "tuple": tuple, "reversed": reversed, "any": any, "all": all
        }
        
        exec(code, {"__builtins__": safe_builtins}, local_env)
        
        modified_df = local_env.get('df')
        if modified_df is not None and not modified_df.equals(df):
             return {"response": {"execution_results": stdout_buffer.getvalue()}, "modified_df": modified_df}

    except Exception as e:
        logger.error(f"   ✗ Code execution failed: {str(e)}")
        return {
            "response": {"execution_results": stdout_buffer.getvalue()},
            "error": str(e)
        }

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

class ModelResponse(BaseModel):
    text_explanation: str
    code_generated: str
    should_execute : bool

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
    past_history_openrouter = get_chat_history_openrouter(sid, limit=10)

    """Build system content for both providers"""
    system_content = f"""{SYSTEM_INSTRUCTION}

### Context of the User's Data
{ai_context}

### CRITICAL: Response Format
You MUST respond with a valid JSON array containing exactly one object with this structure:
[{{"text_explanation": "your explanation here", "code_generated": "python code if needed", "should_execute": true_or_false}}]

Do NOT include any text before or after the JSON. Only output valid JSON."""

    # ==================== HELPER FUNCTIONS ====================
    
    def try_openrouter() -> dict:
        """Attempt to get response from OpenRouter API"""
        messages = [{"role": "system", "content": system_content}]
        messages.extend(past_history_openrouter)
        messages.append({"role": "user", "content": req.message})
        
        logger.info(f"   🌐 Trying OpenRouter ({OPENROUTER_MODEL})...")
        
        response = requests.post(
            url=OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "X-Title": "Quiksight"
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            },
            timeout=60
        )
        
        # Check for rate limits or server errors (trigger fallback)
        if response.status_code in [429, 500, 502, 503, 504]:
            raise requests.exceptions.RequestException(
                f"OpenRouter returned {response.status_code}: {response.text[:200]}"
            )
        
        response.raise_for_status()
        result = response.json()
        
        ai_content = result['choices'][0]['message']['content']
        return {"content": ai_content, "provider": "OpenRouter"}
    
    
    def try_gemini() -> dict:
        """Fallback to Gemini API directly"""
        if not gemini_client:
            raise Exception("Gemini fallback not available - no API key configured")
        
        logger.info("   🔄 Falling back to Gemini API...")
        
        # Get Gemini-formatted history
        past_history_gemini = get_chat_history(sid, limit=10)
        
        chat = gemini_client.chats.create(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_content,
                response_mime_type="application/json",
                response_schema=list[ModelResponse],
                temperature=0.0
            ),
            history=past_history_gemini
        )
        
        response = chat.send_message(req.message)
        return {"content": response.text, "provider": "Gemini"}
    
    
    def parse_ai_response(content: str) -> dict:
        """Parse AI response from either provider"""
        # Clean possible markdown formatting
        text = content.replace('```json', '').replace('```', '').strip()
        
        # Handle both array and object responses
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed[0] if parsed else {}
        return parsed

    # ==================== MAIN LOGIC WITH FALLBACK ====================
    
    try:
        # Try OpenRouter first (primary)
        openrouter_error = None
        ai_result = None
        
        if OPENROUTER_API_KEY:
            try:
                ai_result = try_openrouter()
                logger.info(f"   ✓ OpenRouter responded successfully")
            except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                openrouter_error = str(e)
                logger.warning(f"   ⚠ OpenRouter failed: {openrouter_error[:100]}")
        else:
            openrouter_error = "No OpenRouter API key configured"
            logger.info("   ⚠ OpenRouter not configured, using Gemini")
        
        # Fallback to Gemini if OpenRouter failed
        if ai_result is None:
            try:
                ai_result = try_gemini()
                logger.info(f"   ✓ Gemini responded successfully (fallback)")
            except Exception as gemini_error:
                logger.error(f"   ✗ Gemini fallback also failed: {gemini_error}")
                # If both failed, raise the original OpenRouter error or Gemini error
                if openrouter_error:
                    raise HTTPException(status_code=502, detail=f"All AI providers failed. OpenRouter: {openrouter_error[:100]}")
                raise HTTPException(status_code=502, detail=f"AI service error: {str(gemini_error)[:100]}")
        
        # Save user message after successful AI response
        save_chat_message(sid, "user", req.message)
        
        # Parse AI Response
        try:
            response_data = parse_ai_response(ai_result["content"])
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.error(f"   ✗ Failed to parse AI response: {e}")
            logger.error(f"   Raw content: {ai_result['content'][:500]}")
            response_data = {
                "text_explanation": "I processed your request but had trouble formatting the response.",
                "code_generated": "",
                "should_execute": False
            }
        
        ai_text = response_data.get('text_explanation', '')
        code = response_data.get('code_generated', '')
        should_execute = response_data.get('should_execute', False)
        
        # Save AI text (for model context / history)
        save_chat_message(sid, "model", ai_text)

        execution_results = ""
        if should_execute and code:
            exec_result = execute_user_code(code, df)
            execution_results = exec_result["response"].get("execution_results", "")
            
            # ===== AUTO-RETRY: If code failed, ask AI to fix it =====
            if "error" in exec_result:
                error_msg = exec_result["error"]
                logger.info(f"   🔄 Auto-retrying: asking AI to fix code error...")
                
                fix_prompt = (
                    f"Your previous code failed with this error: {error_msg}\n\n"
                    f"Failed code:\n```python\n{code}\n```\n\n"
                    f"Fix the code and respond in the same JSON format. "
                    f"Make sure variables are defined, indentation is correct, and the code is self-contained."
                )
                
                try:
                    # Re-use whichever provider worked before
                    if ai_result.get("provider") == "OpenRouter" and OPENROUTER_API_KEY:
                        retry_messages = [{"role": "system", "content": system_content}]
                        retry_messages.extend(past_history_openrouter)
                        retry_messages.append({"role": "user", "content": req.message})
                        retry_messages.append({"role": "assistant", "content": ai_result["content"]})
                        retry_messages.append({"role": "user", "content": fix_prompt})
                        
                        retry_resp = requests.post(
                            url=OPENROUTER_URL,
                            headers={
                                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                                "Content-Type": "application/json",
                                "X-Title": "Quiksight"
                            },
                            json={
                                "model": OPENROUTER_MODEL,
                                "messages": retry_messages,
                                "temperature": 0.0,
                                "response_format": {"type": "json_object"}
                            },
                            timeout=60
                        )
                        retry_resp.raise_for_status()
                        fix_content = retry_resp.json()['choices'][0]['message']['content']
                    else:
                        past_history_gemini = get_chat_history(sid, limit=10)
                        fix_chat = gemini_client.chats.create(
                            model="gemini-2.0-flash",
                            config=types.GenerateContentConfig(
                                system_instruction=system_content,
                                response_mime_type="application/json",
                                response_schema=list[ModelResponse],
                                temperature=0.0
                            ),
                            history=past_history_gemini + [
                                types.Content(role="user", parts=[types.Part(text=req.message)]),
                                types.Content(role="model", parts=[types.Part(text=ai_result["content"])]),
                            ]
                        )
                        fix_response = fix_chat.send_message(fix_prompt)
                        fix_content = fix_response.text
                    
                    # Parse the fixed response
                    fix_data = parse_ai_response(fix_content)
                    fixed_code = fix_data.get('code_generated', '')
                    fixed_text = fix_data.get('text_explanation', '')
                    
                    if fixed_code:
                        logger.info("   🔄 Re-executing fixed code...")
                        exec_result2 = execute_user_code(fixed_code, df)
                        
                        if "error" not in exec_result2:
                            # Success! Use the fixed results
                            logger.info("   ✓ Auto-retry succeeded")
                            execution_results = exec_result2["response"].get("execution_results", "")
                            code = fixed_code
                            if fixed_text:
                                ai_text = fixed_text
                            if "modified_df" in exec_result2:
                                exec_result2["modified_df"].to_parquet(df_path)
                            # Clear the original failed result's modified_df tracking
                            exec_result = exec_result2
                        else:
                            logger.warning(f"   ✗ Auto-retry also failed: {exec_result2['error']}")
                            execution_results = '<div style="color:#ef4444;font-size:0.9rem;margin-top:0.5rem;">Something went wrong while processing your request. Please try rephrasing your question.</div>'
                    else:
                        execution_results = '<div style="color:#ef4444;font-size:0.9rem;margin-top:0.5rem;">Something went wrong while processing your request. Please try rephrasing your question.</div>'
                        
                except Exception as retry_err:
                    logger.warning(f"   ✗ Auto-retry request failed: {retry_err}")
                    execution_results = '<div style="color:#ef4444;font-size:0.9rem;margin-top:0.5rem;">Something went wrong while processing your request. Please try rephrasing your question.</div>'
            # ===== END AUTO-RETRY =====
            
            # Save updated DF if modified
            if "modified_df" in exec_result:
                exec_result["modified_df"].to_parquet(df_path)

        # Save the full rendered output for frontend history restoration
        full_model_html = ""
        if ai_text:
            full_model_html += ai_text
        if execution_results:
            full_model_html += execution_results
        if full_model_html:
            save_chat_message(sid, "model_display", full_model_html)

        return {
            "response": {
                "text": ai_text,
                "code": code,
                "execution_results": execution_results,
                "provider": ai_result.get("provider", "unknown")
            }
        }

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"   ✗ Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


@router.get("/chat/history")
async def chat_history_endpoint(sid: str):
    """Return chat history for frontend rendering on page load."""
    redis_key = f"session:{sid}"
    if not redis_client.exists(redis_key):
        raise HTTPException(status_code=404, detail="Session not found")
    
    history_key = f"history:{sid}"
    raw_history = redis_client.lrange(history_key, 0, -1)  # Get all messages
    
    messages = []
    for item in raw_history:
        msg = json.loads(item)
        role = msg.get("role", "")
        text = msg.get("text", "")
        
        # Skip model messages (they're just for AI context)
        # Use model_display messages for the frontend (they contain full rendered HTML)
        if role == "model":
            continue
        
        if role == "user":
            messages.append({"role": "user", "content": text})
        elif role == "model_display":
            messages.append({"role": "ai", "content": text})
    
    return {"messages": messages}

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