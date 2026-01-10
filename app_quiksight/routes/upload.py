from fastapi import APIRouter, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
import os
import polars as pl
import pandas as pd  # Keep for chat.py compatibility
import io
import logging
from typing import Union, Dict
import uuid
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from api_training2.config import GEMINI_API_KEY
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format='\n%(asctime)s | %(levelname)-8s | %(name)-10s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("UPLOAD")

def log_section(title: str, char: str = "━"):
    line = char * 50
    logger.info(f"\n{line}\n  {title}\n{line}")

router = APIRouter()
templates = Jinja2Templates(directory="app_quiksight/templates")

ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]
session_store = {}
client = genai.Client(api_key=GEMINI_API_KEY)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=2)

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

DataFrameOrDict = Union[pl.DataFrame, Dict[str, pl.DataFrame]]


def read_file_polars(file: UploadFile) -> pl.DataFrame:
    """
    Reads uploaded file using Polars (MUCH faster than pandas).
    Returns a Polars DataFrame.
    """
    logger.info(f"Starting file read (Polars): {file.filename}")
    filename = file.filename.lower()

    if filename.endswith(".csv"):
        logger.info("Reading CSV with Polars")
        file.file.seek(0)
        raw_bytes = file.file.read()
        
        # Try different encodings
        for encoding in ["utf-8", "latin1", "iso-8859-1", "cp1252"]:
            try:
                # Polars reads from bytes directly
                df = pl.read_csv(
                    io.BytesIO(raw_bytes),
                    encoding=encoding,
                    ignore_errors=True,
                    infer_schema_length=10000,
                    try_parse_dates=True,
                )
                logger.info(f"CSV loaded (Polars): {df.height} rows, {df.width} cols [{encoding}]")
                return df
            except Exception as e:
                logger.debug(f"Encoding {encoding} failed: {e}")
                continue
        
        raise ValueError("Unable to decode CSV with any supported encoding")

    # Excel handling
    logger.info("Reading Excel with Polars")
    file.file.seek(0)
    raw = file.file.read()
    
    try:
        # Polars read_excel with fastexcel backend
        df = pl.read_excel(
            io.BytesIO(raw),
            infer_schema_length=10000,
        )
        logger.info(f"Excel loaded (Polars): {df.height} rows, {df.width} cols")
        return df
    except Exception as e:
        raise ValueError(f"Failed to read Excel: {e}")


def make_ai_context_polars(df: pl.DataFrame, filename: str) -> str:
    """
    OPTIMIZED: Lightweight context generation using Polars.
    Polars is ~10x faster for these operations.
    """
    logger.info(f"Building AI context (Polars) for {filename}")
    
    num_rows, num_cols = df.height, df.width
    context_parts = [f"{filename} | {num_rows:,} rows × {num_cols} columns"]
    
    # Limit columns analyzed
    MAX_COLS = 15
    cols_to_analyze = df.columns[:MAX_COLS]
    
    # Use sampling for large datasets (Polars sampling is very fast)
    sample_df = df if num_rows <= 10000 else df.sample(n=min(5000, num_rows), seed=42)
    
    summaries = []
    for col in cols_to_analyze:
        dtype = str(df.schema[col])
        
        # Get null count and unique count in one pass
        null_count = sample_df[col].null_count()
        null_pct = (null_count / sample_df.height) * 100 if sample_df.height > 0 else 0
        unique_count = sample_df[col].n_unique()
        
        if df.schema[col].is_numeric():
            # Fast quantile computation with Polars
            try:
                stats = sample_df.select([
                    pl.col(col).min().alias("min"),
                    pl.col(col).median().alias("median"),
                    pl.col(col).max().alias("max"),
                ]).row(0)
                summaries.append(
                    f"{col} (num) — {unique_count} unique, "
                    f"range: {stats[0]:.3g}–{stats[2]:.3g}, median: {stats[1]:.3g}"
                )
            except:
                summaries.append(f"{col} (num) — {unique_count} unique")
        else:
            # Top values for categorical
            try:
                top_vals = (
                    sample_df.group_by(col)
                    .count()
                    .sort("count", descending=True)
                    .head(2)[col]
                    .to_list()
                )
                top_str = ", ".join(str(v)[:15] for v in top_vals if v is not None)
                summaries.append(f"{col} (cat) — {unique_count} unique, top: [{top_str}]")
            except:
                summaries.append(f"{col} (cat) — {unique_count} unique")
    
    if num_cols > MAX_COLS:
        summaries.append(f"... +{num_cols - MAX_COLS} more columns")
    
    context_parts.append("📝 Columns:\n" + "\n".join(summaries))
    
    # Quick quality check
    total_nulls = sample_df.null_count().sum_horizontal()[0]
    context_parts.append(f"ℹ️ Quality: ~{total_nulls:,} missing values in sample")
    
    # Minimal sample data (convert small sample to dicts)
    sample_cols = cols_to_analyze[:8]
    head_sample = df.select(sample_cols).head(3).to_dicts()
    context_parts.append(f"📄 Sample: {head_sample}")
    
    result = "\n\n".join(context_parts)
    logger.info(f"Context built (Polars): {len(result)} chars")
    return result


async def create_chat_session_async(ai_context: str):
    """Create chat session in background to avoid blocking."""
    logger.info("Creating Gemini chat session...")
    
    # Run in thread pool since genai SDK is synchronous
    loop = asyncio.get_event_loop()
    chat_session = await loop.run_in_executor(
        executor,
        lambda: client.chats.create(
            model="gemini-flash-latest",
            config=types.GenerateContentConfig(
                system_instruction=f"{SYSTEM_INSTRUCTION}\n\n###Context of the User's Data\n{ai_context}",
                response_mime_type="application/json",
                response_schema=list[dict],
                temperature=0.0
            )
        )
    )
    logger.info("Chat session created")
    return chat_session


def initialize_session_data(session_id: str, df_polars: pl.DataFrame, chat_session, file: UploadFile, file_size_bytes: int):
    """
    Store session data (called after redirect for speed).
    Converts Polars DataFrame to Pandas for chat.py compatibility.
    """
    size_kb = file_size_bytes / 1024
    file_size = f"{size_kb:.2f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB"
    
    current_timestamp = datetime.now()
    
    # Convert Polars to Pandas for chat.py execution compatibility
    df_pandas = df_polars.to_pandas()
    
    session_store[session_id] = {
        "df": df_pandas,  # Pandas DataFrame for chat.py
        "chat_session": chat_session,
        "file_name": file.filename,
        "file_size": file_size,
        "upload_date": current_timestamp.strftime("%Y-%m-%d"),
        "upload_time": current_timestamp.strftime("%I:%M %p"),
        "columns": df_polars.columns,
        "preview_rows": df_polars.head(5).to_dicts()
    }
    logger.info(f"Session {session_id[:8]} fully initialized (Polars→Pandas converted)")


@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    log_section("📤 NEW FILE UPLOAD REQUEST")

    if not file or file.filename == "":
        logger.warning("No file provided")
        return templates.TemplateResponse("home.html", {"request": request, "error": "Please upload a file"})
    
    logger.info(f"File: {file.filename}")
    
    # Quick size check (non-blocking)
    file.file.seek(0, os.SEEK_END)
    file_size_bytes = file.file.tell()
    file.file.seek(0)
    
    max_size = 30 * 1024 * 1024
    if file_size_bytes > max_size:
        logger.warning("File too large")
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": "File too large. Max 30MB."
        })
    
    # Extension validation
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Invalid extension: {ext}")
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} allowed"
        })

    try:
        pipeline_start = time.time()
        
        # OPTIMIZATION 1: Use Polars for fast file reading
        logger.info("[1/3] Reading file (Polars - async)...")
        df_polars = await run_in_threadpool(read_file_polars, file)
        logger.info(f"✓ Loaded: {df_polars.height} rows × {df_polars.width} cols")
        
        # OPTIMIZATION 2: Fast context generation with Polars
        logger.info("[2/3] Building context (Polars - optimized)...")
        ai_context = await run_in_threadpool(make_ai_context_polars, df_polars, file.filename)
        logger.info(f"✓ Context: {len(ai_context)} chars")
        
        # OPTIMIZATION 3: Create session ID immediately, defer chat creation
        session_id = str(uuid.uuid4())
        logger.info(f"[3/3] Session ID: {session_id[:8]}...")
        
        # Store minimal data first for fast redirect
        session_store[session_id] = {
            "df": df_polars.to_pandas(),  # Convert to Pandas for chat.py
            "chat_session": None,  # Will be created in background
            "file_name": file.filename,
            "file_size": f"{file_size_bytes/1024:.2f} KB",
            "status": "initializing"
        }
        
        # Create chat session in background AFTER redirect
        async def finalize_session():
            try:
                chat_session = await create_chat_session_async(ai_context)
                session_store[session_id]["chat_session"] = chat_session
                session_store[session_id]["status"] = "ready"
                initialize_session_data(session_id, df_polars, chat_session, file, file_size_bytes)
                logger.info(f"Session {session_id[:8]} fully initialized")
            except Exception as e:
                logger.error(f"Background init failed: {e}")
                session_store[session_id]["status"] = "error"
        
        # Schedule background task
        asyncio.create_task(finalize_session())
        
        total_time = time.time() - pipeline_start
        logger.info(f"✅ UPLOAD COMPLETE (Polars): {total_time:.2f}s (async finalization in progress)")
        
        # Redirect immediately (chat page will poll for readiness)
        return RedirectResponse(url=f"/chat?sid={session_id}", status_code=303)

    except Exception as e:
        log_section("❌ UPLOAD FAILED")
        logger.error(f"{type(e).__name__}: {str(e)}")
        return templates.TemplateResponse("home.html", {"request": request, "error": str(e)})
