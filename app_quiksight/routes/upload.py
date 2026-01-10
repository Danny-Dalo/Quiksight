from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool # Essential

import os
import pandas as pd
import io, csv
import numpy as np
import logging
from typing import Union, Dict
import uuid
import time
import gc # Garbage collection

from api_training2.config import GEMINI_API_KEY
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-10s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("UPLOAD")

def log_section(title: str, char: str = "━"):
    line = char * 40
    logger.info(f"\n{line}\n  {title}\n{line}")

router = APIRouter()
templates = Jinja2Templates(directory="app_quiksight/templates")

ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]
# Hard limit for stats calculation to prevent CPU freezing
MAX_STATS_ROWS = 5000 

session_store = {}
client = genai.Client(api_key=GEMINI_API_KEY)

# ================= OPTIMIZED CONTEXT BUILDER =================
def make_ai_context(df: Union[pd.DataFrame, Dict[str, pd.DataFrame]], filename: str) -> str:
    # If dict (multi-sheet), process first sheet only to save memory
    if isinstance(df, dict):
        first_sheet = list(df.keys())[0]
        logger.info(f"Multi-sheet detected. Using first sheet: {first_sheet}")
        return _build_context_for_df(df[first_sheet], filename)
    return _build_context_for_df(df, filename)

def _build_context_for_df(df: pd.DataFrame, filename: str) -> str:
    """Build context using a SAMPLE of data to save CPU/RAM."""
    logger.info(f"Building context for {filename}...")
    
    # 1. OPTIMIZATION: Use a sample for stats, not the whole DB
    # On 0.5 CPU, running .describe() on 500k rows causes timeouts.
    if len(df) > MAX_STATS_ROWS:
        logger.info(f"Dataset too large ({len(df)} rows). Sampling top {MAX_STATS_ROWS} for AI context.")
        stats_df = df.head(MAX_STATS_ROWS)
        data_note = f"(Stats based on first {MAX_STATS_ROWS} rows)"
    else:
        stats_df = df
        data_note = "(Stats based on full data)"

    context_parts = []
    num_cols = len(df.columns)
    num_rows = len(df)
    
    context_parts.append(f"FILE: {filename} | {num_rows:,} rows X {num_cols} columns {data_note}")

    # 2. Lightweight Column Summaries
    summaries = []
    MAX_COLS = 20 # Limit columns analyzed
    
    for col in df.columns[:MAX_COLS]:
        try:
            # Use stats_df (the small sample) for expensive calculations
            missing_pct = stats_df[col].isna().mean() * 100
            dtype = str(df[col].dtype)

            if pd.api.types.is_numeric_dtype(df[col]):
                # Fast numeric stats on sample
                desc = stats_df[col].describe()
                summary = f"{col} (num): range {desc['min']:.2f}-{desc['max']:.2f}, mean {desc['mean']:.2f}"
            
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                summary = f"{col} (date): {stats_df[col].min()} to {stats_df[col].max()}"
                
            else:
                # Fast unique check on sample
                unique_count = stats_df[col].nunique()
                top_vals = stats_df[col].value_counts().head(3).index.tolist()
                summary = f"{col} (cat): {unique_count} unique in sample. Top: {top_vals}"
            
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"{col}: (Error generating stats)")
            
    if num_cols > MAX_COLS:
        summaries.append(f"...and {num_cols - MAX_COLS} more columns.")

    context_parts.append("COLUMNS:\n" + "\n".join(summaries))
    context_parts.append(f"SAMPLE DATA:\n{df.head(3).to_dict(orient='records')}")
    
    return "\n\n".join(context_parts)

SYSTEM_INSTRUCTION = """
You are Quiksight... (Keep your existing system instruction here)
"""

from pydantic import BaseModel
class ModelResponse(BaseModel):
    text_explanation: str
    code_generated: str
    should_execute : bool

# ================= OPTIMIZED FILE READER =================
def read_file(file: UploadFile) -> Union[pd.DataFrame, Dict]:
    logger.info(f"Reading file: {file.filename}")
    filename = file.filename.lower()
    
    # 3. OPTIMIZATION: Stream file directly to pandas. 
    # NEVER use file.read() into a variable on 1GB RAM.
    
    file.file.seek(0) 

    if filename.endswith(".csv"):
        # Try standard UTF-8 first
        try:
            return pd.read_csv(file.file, engine="python", encoding='utf-8')
        except UnicodeDecodeError:
            file.file.seek(0)
            return pd.read_csv(file.file, engine="python", encoding='latin1')

    # Excel Optimization
    # If possible, users should install 'calamine' engine for speed, but default is fine if we don't copy memory
    try:
        # Load Excel directly from the spool file, no memory copy
        return pd.read_excel(file.file)
    except Exception as e:
        logger.error(f"Excel read error: {e}")
        raise ValueError("Could not read Excel file.")

# ================= MAIN ENDPOINT =================
@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    log_section("📤 UPLOAD REQUEST")

    if not file or not file.filename:
        return templates.TemplateResponse("home.html", {"request": request, "error": "No file selected"})

    # Check size - standard lightweight check
    file.file.seek(0, os.SEEK_END)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)
    
    if size_mb > 30:
        return templates.TemplateResponse("home.html", {"request": request, "error": "File > 30MB"})

    try:
        pipeline_start = time.time()
        
        # 1. READ (Threaded + Optimized Memory)
        logger.info("[1/4] Reading file...")
        # We pass the function reference, NOT the result
        df = await run_in_threadpool(read_file, file)
        
        # 2. CONTEXT (Threaded + Optimized CPU)
        logger.info("[2/4] Building context...")
        ai_context = await run_in_threadpool(make_ai_context, df, file.filename)
        
        # 3. CHAT SESSION (Threaded Network Call)
        logger.info("[3/4] Init AI Session...")
        def create_session():
            return client.chats.create(
                model="gemini-flash-latest",
                config=types.GenerateContentConfig(
                    system_instruction=f"{SYSTEM_INSTRUCTION}\n\nDATA CONTEXT:\n{ai_context}",
                    response_mime_type="application/json",
                    response_schema=list[ModelResponse], 
                    temperature=0.0
                )
            )
        chat_session = await run_in_threadpool(create_session)

        # 4. STORE
        session_id = str(uuid.uuid4())
        session_store[session_id] = {
            "df": df, 
            "chat_session": chat_session,
            "file_name": file.filename,
            "columns": list(df.columns),
            "preview_rows": df.head(5).to_dict(orient="records") # Small preview only
        }
        
        # 5. GARBAGE COLLECTION
        # Force cleanup of any temp objects created during read
        gc.collect()

        logger.info(f"✅ DONE in {time.time() - pipeline_start:.2f}s")
        return RedirectResponse(url=f"/chat?sid={session_id}", status_code=303)

    except Exception as e:
        logger.error(f"❌ CRASH: {e}")
        return templates.TemplateResponse("home.html", {"request": request, "error": f"Error: {str(e)}"})