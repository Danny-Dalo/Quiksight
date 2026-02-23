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



def auto_detect_plotly(df: pd.DataFrame):
    """Inspect a DataFrame and return the best Plotly chart as a dict {data, layout}, or None."""
    if df.empty or len(df.columns) < 2:
        return None

    # Classify columns
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or str(df[c].dtype) == 'category']
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    # Convert Period columns to timestamp
    for c in df.columns:
        if hasattr(df[c], 'dt') and hasattr(df[c].dt, 'to_timestamp'):
            try:
                df[c] = df[c].dt.to_timestamp()
                date_cols.append(c)
            except Exception:
                pass

    if not num_cols:
        return None

    try:
        # Case 1: Date + Numeric → Line chart
        if date_cols and num_cols:
            x_col, y_col = date_cols[0], num_cols[0]
            trace = {
                "type": "scatter", "mode": "lines+markers",
                "x": df[x_col].astype(str).tolist(),
                "y": df[y_col].tolist(),
                "name": y_col,
                "line": {"color": "#0ea5e9", "width": 2},
                "marker": {"size": 4}
            }
            layout = {
                "xaxis": {"title": x_col}, "yaxis": {"title": y_col},
                "margin": {"l": 50, "r": 20, "t": 30, "b": 50},
                "height": 340, "plot_bgcolor": "#fff", "paper_bgcolor": "#fff"
            }
            return {"data": [trace], "layout": layout}

        # Case 2: Categorical + Numeric → Horizontal bar
        if cat_cols and num_cols:
            x_col, y_col = cat_cols[0], num_cols[0]
            sorted_df = df.sort_values(y_col, ascending=True).tail(15)
            trace = {
                "type": "bar", "orientation": "h",
                "y": sorted_df[x_col].astype(str).tolist(),
                "x": sorted_df[y_col].tolist(),
                "name": y_col,
                "marker": {"color": "#0ea5e9"}
            }
            layout = {
                "xaxis": {"title": y_col}, "yaxis": {"title": "", "automargin": True},
                "margin": {"l": 10, "r": 20, "t": 30, "b": 50},
                "height": max(260, len(sorted_df) * 28),
                "plot_bgcolor": "#fff", "paper_bgcolor": "#fff"
            }
            return {"data": [trace], "layout": layout}

        # Case 3: Multiple numeric columns → grouped bar using index/first col as labels
        if len(num_cols) >= 2:
            label_col = cat_cols[0] if cat_cols else df.columns[0]
            traces = []
            colors = ["#0ea5e9", "#f97316", "#8b5cf6", "#10b981", "#ef4444"]
            for i, nc in enumerate(num_cols[:5]):
                traces.append({
                    "type": "bar",
                    "x": df[label_col].astype(str).tolist(),
                    "y": df[nc].tolist(),
                    "name": nc,
                    "marker": {"color": colors[i % len(colors)]}
                })
            layout = {
                "barmode": "group",
                "xaxis": {"title": str(label_col)},
                "margin": {"l": 50, "r": 20, "t": 30, "b": 50},
                "height": 340, "plot_bgcolor": "#fff", "paper_bgcolor": "#fff"
            }
            return {"data": traces, "layout": layout}

        # Case 4: Fallback — simple vertical bar of first numeric column
        y_col = num_cols[0]
        labels = df.iloc[:, 0].astype(str).tolist()[:20]
        trace = {
            "type": "bar",
            "x": labels,
            "y": df[y_col].tolist()[:20],
            "marker": {"color": "#0ea5e9"}
        }
        layout = {
            "yaxis": {"title": y_col},
            "margin": {"l": 50, "r": 20, "t": 30, "b": 50},
            "height": 340, "plot_bgcolor": "#fff", "paper_bgcolor": "#fff"
        }
        return {"data": [trace], "layout": layout}

    except Exception as e:
        logger.warning(f"auto_detect_plotly failed: {e}")
        return None


def dataframe_to_styled_html(df: pd.DataFrame, download_id: str = None, max_rows=10):
    """Convert a DataFrame into a tabbed Chart + Table HTML container."""
    viz_id = str(uuid.uuid4())[:8]
    df_preview = df.head(max_rows)

    # ── Build table HTML ──
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
                cell = '<span class="qs-na">N/A</span>'
            elif len(str(val)) > 60:
                safe_val = str(val).replace('"', '&quot;')
                cell = f'<span title="{safe_val}" style="cursor:help">{safe_val[:57]}…</span>'
            else:
                cell = str(val)
            tbody += f"<td>{cell}</td>"
        tbody += "</tr>"
    tbody += "</tbody>"

    table_html = f"""<div class="overflow-x-auto">
        <table class="qs-table">{thead}{tbody}</table>
    </div>
    <div class="qs-row-info">Showing {len(df_preview)} of {len(df)} rows</div>"""

    # ── Build chart JSON ──
    chart_json = auto_detect_plotly(df_preview)
    chart_panel_content = ""
    if chart_json:
        chart_json_str = json.dumps(chart_json)
        chart_panel_content = f"""<div class="qs-plotly-chart" id="chart-{viz_id}"></div>
            <script class="qs-plotly-data" type="application/json">{chart_json_str}</script>"""
    else:
        chart_panel_content = '<p style="color:#9ca3af;text-align:center;padding:2rem;">No chart available for this data</p>'

    # ── Download button ──
    download_btn = ""
    if download_id:
        download_btn = f"""<a href="/chat/download/{download_id}" target="_blank" class="qs-download-btn">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
            </svg>CSV</a>"""

    # ── Assemble tabbed container ──
    return f"""<div class="qs-viz" id="viz-{viz_id}">
    <div class="qs-tab-bar">
        <button class="qs-tab qs-tab-active" onclick="qsTabSwitch(this,'chart-panel-{viz_id}')">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="#4d4f4e" class="size-4">
            <path d="M12 2a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h1a1 1 0 0 0 1-1V3a1 1 0 0 0-1-1h-1ZM6.5 6a1 1 0 0 1 1-1h1a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1h-1a1 1 0 0 1-1-1V6ZM2 9a1 1 0 0 1 1-1h1a1 1 0 0 1 1 1v4a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V9Z" />
        </svg>
        </button>
        <button class="qs-tab" onclick="qsTabSwitch(this,'table-panel-{viz_id}')">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="#4d4f4e" class="size-4">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 0 1-1.125-1.125M3.375 19.5h7.5c.621 0 1.125-.504 1.125-1.125m-9.75 0V5.625m0 12.75v-1.5c0-.621.504-1.125 1.125-1.125m18.375 2.625V5.625m0 12.75c0 .621-.504 1.125-1.125 1.125m1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125m0 3.75h-7.5A1.125 1.125 0 0 1 12 18.375m9.75-12.75c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125m19.5 0v1.5c0 .621-.504 1.125-1.125 1.125M2.25 5.625v1.5c0 .621.504 1.125 1.125 1.125m0 0h17.25m-17.25 0h7.5c.621 0 1.125.504 1.125 1.125M3.375 8.25c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125m17.25-3.75h-7.5c-.621 0-1.125.504-1.125 1.125m8.625-1.125c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125m-17.25 0h7.5m-7.5 0c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125M12 10.875v-1.5m0 1.5c0 .621-.504 1.125-1.125 1.125M12 10.875c0 .621.504 1.125 1.125 1.125m-2.25 0c.621 0 1.125.504 1.125 1.125M13.125 12h7.5m-7.5 0c-.621 0-1.125.504-1.125 1.125M20.625 12c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125m-17.25 0h7.5M12 14.625v-1.5m0 1.5c0 .621-.504 1.125-1.125 1.125M12 14.625c0 .621.504 1.125 1.125 1.125m-2.25 0c.621 0 1.125.504 1.125 1.125m0 1.5v-1.5m0 0c0-.621.504-1.125 1.125-1.125m0 0h7.5" />
        </svg>
        </button>
        {download_btn}
    </div>
    <div class="qs-panel qs-panel-active" id="chart-panel-{viz_id}" style="display:block">
        {chart_panel_content}
    </div>
    <div class="qs-panel" id="table-panel-{viz_id}" style="display:none">
        {table_html}
    </div>
</div>"""




# ==============================================================================
#                           CODE EXECUTION
# ==============================================================================
def execute_user_code(code: str, df: pd.DataFrame):
    """Execute AI-generated code in a sandboxed environment."""
    logger.info("   Executing AI-generated code...")
    
    # ── Strip import statements (AI sometimes ignores "NO imports" rule) ──
    clean_lines = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            logger.warning(f"   ⚠ Stripped import line: {stripped[:80]}")
            continue
        clean_lines.append(line)
    code = "\n".join(clean_lines)
    
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

    import datetime as _dt, math as _math, re as _re, collections as _collections

    local_env = {
        "df": df.copy(), "pd": pd, "np": np, "display_table": display_table,
        # Pre-loaded modules so AI doesn't need import statements
        "datetime": _dt, "math": _math, "re": _re, "json": json, "collections": _collections,
    }
    stdout_buffer = io.StringIO()
    sys.stdout = stdout_buffer

    try:
        # Whitelisted built-ins for safety
        safe_builtins = {
            "int": int, "float": float, "str": str, "bool": bool, "list": list, "dict": dict,
            "len": len, "range": range, "enumerate": enumerate, "zip": zip,
            "sorted": sorted, "sum": sum, "min": min, "max": max, "print": print, "type": type,
            "round": round, "abs": abs, "isinstance": isinstance, "tuple": tuple, "set": set,
            "map": map, "filter": filter, "any": any, "all": all, "hasattr": hasattr,
            "getattr": getattr, "reversed": reversed, "iter": iter, "next": next,
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