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
from api_training2.config import GEMINI_API_KEY
from google import genai
from google.genai import types
import csv
import time
from typing import Union, Dict

client = genai.Client(api_key=GEMINI_API_KEY)

# Return type
DataFrameOrDict = Union[pd.DataFrame, Dict[str, pd.DataFrame]] 


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
    status = session_data.get("status", "ready")
    logger.info(f"   ✓ Session found: {session_data['file_name']} (Status: {status})")
    
    # Ensure downloads dict exists (backward compatibility)
    if "downloads" not in session_data:
        session_data["downloads"] = {}
        
    # Safe access for pending sessions
    df = session_data.get("df")
    num_rows = len(df) if df is not None else 0
    num_cols = len(df.columns) if df is not None else 0
    columns = session_data.get("columns", [])
    preview = session_data.get("preview_rows", [])

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "session_id": sid,
        "status": status,
        "file_name": session_data["file_name"],
        "file_size": session_data["file_size"],
        "file_extension": os.path.splitext(session_data["file_name"])[1],
        "upload_date": session_data["upload_date"],
        "upload_time": session_data["upload_time"],
        "num_rows": num_rows,
        "num_columns": num_cols,
        "columns": columns,
        "preview_rows": preview,
        "cache_buster": datetime.datetime.now().timestamp()
    })






SYSTEM_INSTRUCTION = """
You are Quiksight's data assistant—a friendly, sharp analyst who helps users understand their data conversationally.

PERSONALITY
- Speak like a helpful colleague, not a robot or corporate chatbot
- Use contractions naturally (you've, there's, I'll)
- Be direct—skip preambles like "Sure!" or "Great question!"
- Match the user's tone and detail level
- Stay focused on the dataset; gently redirect off-topic questions

RESPONSE FORMAT
Use clean, minimal HTML:
- <p> for paragraphs
- <strong> for emphasis (sparingly)
- Lists when appropriate:
  <ul class="list-disc list-inside space-y-1 mt-2 mb-2"><li>Item</li></ul>
  <ol class="list-decimal list-inside space-y-1 mt-2 mb-2"><li>Step</li></ol>

CRITICAL RULE: TEXT + CODE MUST BLEND SEAMLESSLY
Your text_explanation and code output appear as ONE message to the user.

Since you write text_explanation BEFORE code runs, you CANNOT know computed values.

DO: Leave text_explanation empty when code computes the answer. Let code print the full response.
DON'T: Write numbers/values in text_explanation—you'll hallucinate wrong data.

PATTERN A — Computed Values (numbers, counts, aggregations):
text_explanation: ""
code_generated: |
  total = df['Sales'].sum()
  print(f"<p>Total sales: <strong>${total:,.2f}</strong></p>")

PATTERN B — Tables or Lists from Data:
text_explanation: "<p>Here's a breakdown by region:</p>"
code_generated: |
  result = df.groupby('Region')['Sales'].sum().reset_index()
  result.columns = ['Region', 'Total Sales']
  display_table(result)

PATTERN C — General Questions (no computation needed):
text_explanation: "<p>The dataset contains customer orders with columns for date, product, quantity, and price.</p>"
code_generated: ""
should_execute: false

CODE RULES
- Available: df (DataFrame), pd, np, display_table()
- NO imports, NO file I/O
- For DataFrames: use display_table(df), NOT print()
- Always flatten MultiIndex after groupby:
  CORRECT: df.groupby('X').size().reset_index(name='Count')
  WRONG: df.groupby('X').agg({'Y': ['count']})
- Format numbers nicely: {:,} for thousands, :.2f for decimals
- Wrap risky operations in try-except

WHEN TO EXECUTE CODE
should_execute: true → calculations, aggregations, filtering, transformations, showing data subsets
should_execute: false → explaining structure, describing columns, interpretation without computation

SEAMLESS OUTPUT
Your response should feel like natural conversation. Never mention "executing code" or "running analysis"—just present the answer as if you knew it all along.
"""

# ================= HELPER FUNCTIONS MOVED FROM UPLOAD.PY =================

def make_ai_context(df: DataFrameOrDict, filename: str, sample_size: int = 5) -> str:
    logger.info(f"Building AI context for file: {filename}")
    if isinstance(df, pd.DataFrame):
        logger.info(f"Processing single DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return _build_context_for_df(df, filename, sample_size)
    else:
        # Multi-sheet: Build context for each
        logger.info(f"Processing multi-sheet file with {len(df)} sheets: {list(df.keys())}")
        contexts = []
        for sheet_name, sheet_df in df.items():
            logger.info(f"Building context for sheet: {sheet_name}")
            contexts.append(f"📑 Sheet: {sheet_name}\\n" + _build_context_for_df(sheet_df, filename, sample_size))
        return "\n\n---\n\n".join(contexts)


def _build_context_for_df(df: pd.DataFrame, filename: str, sample_size: int) -> str:
    """Build a token-efficient context summary for the AI."""
    logger.debug(f"Starting context build for {filename}")
    context_parts = []
    num_cols = len(df.columns)
    num_rows = len(df)
    logger.info(f"DataFrame stats - Rows: {num_rows:,}, Columns: {num_cols}")
    
    # Config for token efficiency
    MAX_COLS_DETAILED = 25  # Full stats for first N columns
    MAX_SAMPLE_COLS = 12    # Columns to include in sample data
    MAX_RAND_SAMPLES = 3    # Random sample rows
    
    # 1. File-level metadata (compact)
    context_parts.append(f"{filename} | {num_rows:,} rows X {num_cols} columns")

    # 2. Column summaries (limited)
    summaries = []
    cols_to_detail = list(df.columns[:MAX_COLS_DETAILED])
    remaining_cols = num_cols - len(cols_to_detail)
    
    for col in cols_to_detail:
        dtype = str(df[col].dtype)
        missing_pct = df[col].isna().mean() * 100
        unique_vals = df[col].nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe(percentiles=[.25, .5, .75])
            # Compact numeric summary
            col_summary = (
                f"{col} (num) — {unique_vals} unique, "
                f"range: {desc['min']:.4g}–{desc['max']:.4g}, "
                f"mean: {desc['mean']:.4g}, missing: {missing_pct:.0f}%"
            )
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_summary = (
                f"{col} (date) — {unique_vals} unique, "
                f"range: {df[col].min()} → {df[col].max()}, missing: {missing_pct:.0f}%"
            )
        else:  # categorical or text
            top_vals = list(df[col].value_counts(dropna=True).head(3).index)
            top_str = ", ".join(str(v)[:20] for v in top_vals)  # Truncate long values
            col_summary = (
                f"{col} (cat) — {unique_vals} unique, "
                f"top: [{top_str}], missing: {missing_pct:.0f}%"
            )
        summaries.append(col_summary)
    
    if remaining_cols > 0:
        summaries.append(f"... and {remaining_cols} more columns")
    
    context_parts.append("📝 Columns:\n" + "\n".join(summaries))

    # 3. Data quality summary (one line)
    total_missing = df.isna().sum().sum()
    dup_count = df.duplicated().sum()
    context_parts.append(f" Quality: {total_missing:,} missing values ({df.isna().mean().mean()*100:.1f}%), {dup_count:,} duplicates")

    # 4. Structural insights (only if issues detected)
    empty_rows = df.isnull().all(axis=1).sum()
    empty_cols = df.isnull().all(axis=0).sum()
    
    if empty_rows > 0 or empty_cols > 0:
        context_parts.append(f" Structure: {empty_rows} empty rows, {empty_cols} empty columns")
    
    # 5. Sample data (truncated for wide datasets)
    sample_cols = list(df.columns[:MAX_SAMPLE_COLS])
    df_sample = df[sample_cols]
    
    head_sample = df_sample.head(2).to_dict(orient="records")
    context_parts.append(f" First rows: {head_sample}")
    
    # Only add random sample if dataset is larger than head sample
    if num_rows > 5:
        rand_sample = df_sample.sample(min(MAX_RAND_SAMPLES, num_rows), random_state=42).to_dict(orient="records")
        context_parts.append(f" Random sample: {rand_sample}")
    
    if num_cols > MAX_SAMPLE_COLS:
        context_parts.append(f"(Sample shows first {MAX_SAMPLE_COLS} of {num_cols} columns)")

    return "\n\n".join(context_parts)

def read_file(file_path: str, filename: str) -> DataFrameOrDict:
    """Read file from disk (CSV/Excel)."""
    logger.info(f"Reading file from disk: {file_path}")
    
    if filename.lower().endswith(".csv"):
        for encoding in ["utf-8", "latin1", "iso-8859-1", "cp1252"]:
            try:
                df = pd.read_csv(file_path, encoding=encoding, engine="python", skip_blank_lines=True)
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError("Unable to decode CSV with supported encodings.")
        
    # Excel
    try:
        xls = pd.ExcelFile(file_path)
        sheets = xls.sheet_names
        if len(sheets) != 1:
            raise ValueError("Only single-sheet Excel files supported.")
        return pd.read_excel(xls, sheet_name=sheets[0])
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")

from .upload import ModelResponse

# =========================================================================

class ChatRequest(BaseModel):
    message: str

def prepare_session(sid: str):
    """Run heavy initialization for a session: read file, build context, init Gemini."""
    logger.info(f"⚠️ Starting LAZY INITIALIZATION for session {sid}")
    start_time = time.time()
    
    session = session_store.get(sid)
    if not session:
        raise ValueError("Session not found")
        
    file_path = session.get("file_path")
    filename = session.get("file_name")
    
    if not file_path or not os.path.exists(file_path):
        raise ValueError(f"File not found on disk: {file_path}")
        
    # 1. Read File
    df = read_file(file_path, filename)
    
    # 2. Build Context
    ai_context = make_ai_context(df, filename)
    
    # 3. Create Chat Session
    chat_session = client.chats.create(
        model="gemini-flash-latest",
        config=types.GenerateContentConfig(
            system_instruction=f"{SYSTEM_INSTRUCTION}\n\n ###Context of the User's Data\n {ai_context}",
            response_mime_type="application/json",
            response_schema=list[ModelResponse], 
            temperature=0.0
        )
    )
    
    # 4. Update Session
    session["df"] = df
    session["chat_session"] = chat_session
    session["columns"] = list(df.columns)
    session["preview_rows"] = df.head(5).to_dict(orient="records")
    session["num_rows"] = len(df)
    session["num_columns"] = len(df.columns)
    session["status"] = "ready"
    
    logger.info(f"✅ Initialization complete ({time.time() - start_time:.2f}s)")


@router.get("/chat/status")
async def chat_status(sid: str):
    """Check session status and trigger lazy load if pending."""
    if sid not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = session_store[sid]
    status = session.get("status", "pending")
    
    if status == "pending":
        try:
            # Synchronous execution (will block this request, but that's what we want for now)
            prepare_session(sid)
            return {"status": "ready"}
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            logger.error(traceback.format_exc())
            return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})
            
    return {"status": "ready"}

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

