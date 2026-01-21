from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import pandas as pd
import io
import logging
from typing import Union, Dict
import uuid
import time, datetime
import json

# Redis Integration
from app_quiksight.storage.redis import redis_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='\n%(asctime)s | %(levelname)-8s | %(name)-10s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("UPLOAD")

# Constants & Setup
DATA_DIR = "data/sessions"
os.makedirs(DATA_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]
MAX_UPLOAD_SIZE = 30 * 1024 * 1024  # 30MB

router = APIRouter()
templates = Jinja2Templates(directory="app_quiksight/templates")

def log_section(title: str, char: str = "━"):
    """section divider for better log readability."""
    line = char * 50
    logger.info(f"\n{line}\n  {title}\n{line}")

# ==============================================================================
#  AI CONTEXT & PROMPT GENERATION (Imported by chat.py)
# ==============================================================================

SYSTEM_INSTRUCTION = """
# You are a friendly, sharp analyst who helps users understand their data conversationally.

# PERSONALITY
# - Speak like a helpful colleague, not a robot or corporate chatbot
# - Use contractions naturally (you've, there's, I'll)
# - Be direct—skip preambles like "Sure!" or "Great question!"
# - Match the user's tone and detail level
# - Stay focused on the dataset; gently redirect off-topic questions

# RESPONSE FORMAT
# Use clean, minimal HTML:
# - <p> for paragraphs
# - <strong> for emphasis (sparingly)
# - Lists when appropriate:
#   <ul class="list-disc list-inside space-y-1 mt-2 mb-2"><li>Item</li></ul>
#   <ol class="list-decimal list-inside space-y-1 mt-2 mb-2"><li>Step</li></ol>

# CRITICAL RULE: TEXT + CODE MUST BLEND SEAMLESSLY
# Your text_explanation and code output appear as ONE message to the user.

# Since you write text_explanation BEFORE code runs, you CANNOT know computed values.

# DO: Leave text_explanation empty when code computes the answer. Let code print the full response.
# DON'T: Write numbers/values in text_explanation—you'll hallucinate wrong data.

# PATTERN A — Computed Values (numbers, counts, aggregations):
# text_explanation: ""
# code_generated: |
#   total = df['Sales'].sum()
#   print(f"<p>Total sales: <strong>${total:,.2f}</strong></p>")

# PATTERN B — Tables or Lists from Data:
# text_explanation: "<p>Here's a breakdown by region:</p>"
# code_generated: |
#   result = df.groupby('Region')['Sales'].sum().reset_index()
#   result.columns = ['Region', 'Total Sales']
#   display_table(result)

# PATTERN C — General Questions (no computation needed):
# text_explanation: "<p>The dataset contains customer orders with columns for date, product, quantity, and price.</p>"
# code_generated: ""
# should_execute: false

# CODE RULES
# - Available: df (DataFrame), pd, np, display_table()
# - NO imports, NO file I/O
# - For DataFrames: use display_table(df), NOT print()
# - Always flatten MultiIndex after groupby:
#   CORRECT: df.groupby('X').size().reset_index(name='Count')
#   WRONG: df.groupby('X').agg({'Y': ['count']})
# - Format numbers nicely: {:,} for thousands, :.2f for decimals
# - Wrap risky operations in try-except

# WHEN TO EXECUTE CODE
# should_execute: true → calculations, aggregations, filtering, transformations, showing data subsets
# should_execute: false → explaining structure, describing columns, interpretation without computation

# SEAMLESS OUTPUT
# Your response should feel like natural conversation. Never mention "executing code" or "running analysis"—just present the answer as if you knew it all along.
"""

def make_ai_context(df: Union[pd.DataFrame, Dict[str, pd.DataFrame]], filename: str, sample_size: int = 5) -> str:
    logger.info(f"Building AI context for file: {filename}")
    if isinstance(df, pd.DataFrame):
        return _build_context_for_df(df, filename, sample_size)
    else:
        # Multi-sheet: Build context for each
        contexts = []
        for sheet_name, sheet_df in df.items():
            contexts.append(f"📑 Sheet: {sheet_name}\n" + _build_context_for_df(sheet_df, filename, sample_size))
        return "\n\n---\n\n".join(contexts)

def _build_context_for_df(df: pd.DataFrame, filename: str, sample_size : int) -> str:
    """Build a token-efficient context summary for the AI."""
    context_parts = []
    num_cols = len(df.columns)
    num_rows = len(df)
    
    # Config for token efficiency
    MAX_COLS_DETAILED = 25  
    MAX_SAMPLE_COLS = 12    
    MAX_RAND_SAMPLES = 3    
    
    # 1. File-level metadata
    context_parts.append(f"{filename} | {num_rows:,} rows X {num_cols} columns")

    # 2. Column summaries
    summaries = []
    cols_to_detail = list(df.columns[:MAX_COLS_DETAILED])
    remaining_cols = num_cols - len(cols_to_detail)
    
    for col in cols_to_detail:
        missing_pct = df[col].isna().mean() * 100
        unique_vals = df[col].nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe(percentiles=[.25, .5, .75])
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
            top_str = ", ".join(str(v)[:20] for v in top_vals) 
            col_summary = (
                f"{col} (cat) — {unique_vals} unique, "
                f"top: [{top_str}], missing: {missing_pct:.0f}%"
            )
        summaries.append(col_summary)
    
    if remaining_cols > 0:
        summaries.append(f"... and {remaining_cols} more columns")
    
    context_parts.append("📝 Columns:\n" + "\n".join(summaries))

    # 3. Data quality & Structure
    total_missing = df.isna().sum().sum()
    dup_count = df.duplicated().sum()
    context_parts.append(f" Quality: {total_missing:,} missing, {dup_count:,} duplicates")

    # 4. Sample data
    sample_cols = list(df.columns[:MAX_SAMPLE_COLS])
    df_sample = df[sample_cols]
    
    head_sample = df_sample.head(2).astype(str).to_dict(orient="records")
    context_parts.append(f" First rows: {head_sample}")
    
    if num_rows > 5:
        rand_sample = df_sample.sample(min(MAX_RAND_SAMPLES, num_rows), random_state=42).astype(str).to_dict(orient="records")
        context_parts.append(f" Random sample: {rand_sample}")

    return "\n\n".join(context_parts)



# ==============================================================================
#                        FILE READING AND VALIDATION
# ==============================================================================

def read_file(file: UploadFile) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Makes sure uploaded files are either csv or excel
    Reads CSV or Excel file into a DataFrame.
    """
    logger.info(f"Starting file read for: {file.filename}")
    filename = file.filename.lower()

    # Reading CSV files
    if filename.endswith(".csv"):
        for encoding in ["utf-8", "latin1", "iso-8859-1", "cp1252"]:
            try:
                file.file.seek(0)
                df = pd.read_csv(file.file, encoding=encoding, engine="python")
                logger.info(f"CSV read success ({encoding})")
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError("Unable to decode CSV with supported encodings.")

    # Excel Handling
    file.file.seek(0)
    raw = file.file.read()
    try:
        xls = pd.ExcelFile(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")

    sheets = xls.sheet_names
    if not sheets: raise ValueError("No sheets found in Excel file.")
    if len(sheets) != 1: raise ValueError("Only single-sheet Excel files are supported at this time.")

    return pd.read_excel(xls, sheet_name=sheets[0])



# ==============================================================================
#                                   UPLOAD ROUTE
# ==============================================================================

@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    log_section("       ==NEW FILE UPLOAD REQUEST==       ")

    """Check if a file is actually uploaded when request is made"""
    if not file or file.filename == "":
        return templates.TemplateResponse("home.html", {"request": request, "error": "Please upload a file"})
    
    # """ Check file size (Maximum 30MB). Replaced with logic in main.py"""
    # file_size_bytes = file.size
    # logger.info(f"   Size: {file_size_bytes / 1024:.2f} KB ({file_size_bytes / (1024*1024):.2f} MB)")

    # if file_size_bytes > MAX_UPLOAD_SIZE:
    #     logger.warning(f" Rejected: File size exceeds 30MB limit")
    #     return templates.TemplateResponse("home.html", {"request": request,"error": "File too large. Maximum size allowed is 30MB."})
    
    """ Checks if the uploaded file's extension is allowed """
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse("home.html", {"request": request, "error": "Invalid file type"})
    
    

    try:
        log_section("    PROCESSING PIPELINE    ", "─")
        pipeline_start = time.time()    # tracking how fast it takes to complete processes
        
        """Read and validate the file"""
        logger.info("\n Step [1/3]  Reading file...")
        df = read_file(file)

        """Summarize data and generate data context for the AI model"""
        logger.info("\n Step [2/3]  Generating AI Context...")
        data_context = make_ai_context(df, file.filename)
        
        """Create a session ID for each uploaded file for reference"""
        logger.info("\n Step [3/3] Creating Session...")
        session_id = str(uuid.uuid4())
        
        """Calculating fiile size(in KB and MB)"""
        size_kb = file.size / 1024
        file_size = f"{size_kb:.2f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB"
        current_timestamp = datetime.datetime.now()

        # Sanitize DF for Parquet (Convert Objects to Strings to avoid errors)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)

        # Save Parquet
        dataframe_path = f"{DATA_DIR}/{session_id}.parquet"
        df.to_parquet(dataframe_path, engine="pyarrow")

        # Prepare Redis Payload
        # We ensure preview rows are strings to avoid JSON errors with timestamps
        preview_data = df.head(5).astype(str).to_dict(orient="records")
        rows = len(df)
        
        session_data = {
            "file_name": file.filename,
            "file_size": file_size,
            "file_extension": os.path.splitext(file.filename)[1],
            "upload_date": str(current_timestamp.strftime("%Y-%m-%d")),
            "upload_time": str(current_timestamp.strftime("%I:%M %p")),
            "columns": json.dumps(list(df.columns)),
            "num_rows" : rows,
            "preview_rows": json.dumps(preview_data),
            "dataframe_path": dataframe_path,
            "data_context": data_context
        }

        """ Redis stores everything necessary to resume the chat session later as long as it has not expired
            redis_key: Unique, used to match session activity (chat_history, file_info, data context) in redis, 
            made from randomly generated session_id
        """
        redis_key = f"session:{session_id}"
        """.hset: sets key, value pairs in a redis hash
            key --> session:{session_id}
            mapping(takes in a dictionary) --> session_data
        """
        redis_client.hset(redis_key, mapping=session_data)
        """Sets session expiry time of redis key(along with its values)"""
        redis_client.expire(redis_key, 3600) # 1 Hour Expiry

        total_time = time.time() - pipeline_start   # How long the whole upload process took
        log_section("               UPLOAD COMPLETE     ", "─")
        logger.info(f"   Session: {session_id[:8]}... | Time: {total_time:.2f}s")
        
        return RedirectResponse(url=f"/chat?sid={session_id}", status_code=303)

    except Exception as e:
        log_section("                UPLOAD FAILED      ", "═")
        logger.error(f"   Error: {str(e)}")
        return templates.TemplateResponse("home.html", {"request": request, "error": str(e)})
