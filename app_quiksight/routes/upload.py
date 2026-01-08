
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
import os
import pandas as pd
import io, csv
import numpy as np
import logging

from typing import Union, Dict

from api_training2.config import GEMINI_API_KEY
import uuid
import time

from google import genai
from google.genai import types

# Configure logging with cleaner format
logging.basicConfig(
    level=logging.INFO,
    format='\n%(asctime)s | %(levelname)-8s | %(name)-10s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("UPLOAD")


def log_section(title: str, char: str = "━"):
    """Log a visual section divider for better readability."""
    line = char * 50
    logger.info(f"\n{line}\n  {title}\n{line}")

router = APIRouter()
templates = Jinja2Templates(directory="app_quiksight/templates")

ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

# In-memory storage (will replace with DB/Redis)
session_store = {}

client = genai.Client(api_key=GEMINI_API_KEY)


# ========== Need to review these functions ============
def make_ai_context(df: Union[pd.DataFrame, Dict[str, pd.DataFrame]], filename: str, sample_size: int = 5) -> str:
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
            contexts.append(f"📑 Sheet: {sheet_name}\n" + _build_context_for_df(sheet_df, filename, sample_size))
        return "\n\n---\n\n".join(contexts)


def _build_context_for_df(df: pd.DataFrame, filename: str, sample_size: int) -> str:
    """Build a token-efficient context summary for the AI."""
    logger.debug(f"Starting context build for {filename}")
    context_parts = []
    num_cols = len(df.columns)
    num_rows = len(df)
    logger.info(f"DataFrame stats - Rows: {num_rows:,}, Columns: {num_cols}")
    logger.debug(f"Column names: {list(df.columns)}")
    logger.debug(f"Data types: {df.dtypes.to_dict()}")
    
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
        logger.warning(f"Data quality issue detected - Empty rows: {empty_rows}, Empty columns: {empty_cols}")
        context_parts.append(f" Structure: {empty_rows} empty rows, {empty_cols} empty columns")
    
    # Check for multiple data blocks (only report if fragmented)
    non_empty_mask = ~df.isnull().all(axis=1)
    block_starts = np.where(non_empty_mask & ~non_empty_mask.shift(fill_value=False))[0]
    if len(block_starts) > 1:
        logger.warning(f"Data fragmentation detected - {len(block_starts)} separate data blocks found")
        context_parts.append(f" Data appears fragmented into {len(block_starts)} sections")

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

    logger.info(f"AI context built successfully for {filename} ({len('\n\n'.join(context_parts))} chars)")
    return "\n\n".join(context_parts)
# =======================================

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






# Return type because the return type of the function can be a dictionary of file sheets as well
DataFrameOrDict = Union[pd.DataFrame, Dict[str, pd.DataFrame]] 


def read_file(file: UploadFile) -> DataFrameOrDict:
    """
    Reads an uploaded file (CSV or Excel) and returns a DataFrame
    or a dict of DataFrames if file has multiple sheets(feature not yet added).
    """
    logger.info(f"Starting file read operation for: {file.filename}")
    filename = file.filename.lower()

    # ---- CSV Handling ----
    if filename.endswith(".csv"):
        logger.info("Detected CSV file format")
        for encoding in ["utf-8", "latin1", "iso-8859-1", "cp1252"]:
            try:
                logger.debug(f"Attempting to read CSV with encoding: {encoding}")
                file.file.seek(0)
                # when file is read, pointer reads it and goes to the end (like when you read a book)
                # If a previous read failed, the cursor is already at the end so it would be seen as empty if you try again
                # .seek() resets it back to the beginning (going back to the beginning of the book) to read again
                df = pd.read_csv(file.file,
                                   encoding=encoding,
                                   engine="python",
                                   quotechar='"',
                                   quoting=csv.QUOTE_MINIMAL,
                                   skip_blank_lines=True,
                                   )
                logger.info(f"CSV file read successfully with encoding: {encoding}")
                logger.info(f"Loaded DataFrame: {len(df)} rows, {len(df.columns)} columns")
                return df
            except UnicodeDecodeError:
                logger.debug(f"Encoding {encoding} failed, trying next...")
                continue
        logger.error("Failed to decode CSV with any supported encoding")
        raise ValueError("Unable to decode CSV with supported encodings.")

    # ---- Excel Handling ----
    logger.info("Detected Excel file format")
    file.file.seek(0)  # takes pointer to beginning of the file
    raw = file.file.read()
    logger.debug(f"Read {len(raw)} bytes from Excel file")

    try:
        xls = pd.ExcelFile(io.BytesIO(raw))
        logger.info(f"Excel file opened successfully")
    except Exception as e:
        logger.error(f"Failed to open Excel file: {e}")
        raise ValueError(f"Failed to read Excel file: {e}")

    sheets = xls.sheet_names
    logger.info(f"Found {len(sheets)} sheet(s): {sheets}")
    
    if not sheets:
        logger.error("No sheets found in Excel file")
        raise ValueError("No sheets found in Excel file.")

    if len(sheets) != 1:
        logger.warning(f"Multiple sheets detected ({len(sheets)}), but only single sheet is supported")
        raise ValueError("Only Excel files with a single sheet are supported at this time.")

    df = pd.read_excel(xls, sheet_name=sheets[0])
    logger.info(f"Excel file read successfully from sheet: {sheets[0]}")
    logger.info(f"Loaded DataFrame: {len(df)} rows, {len(df.columns)} columns")
    return df





from pydantic import BaseModel
class ModelResponse(BaseModel):
    text_explanation: str
    code_generated: str
    should_execute : bool


# Uploaded files are validated and submitted to the chat endpoint for the AI model to use
@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    log_section("📤 NEW FILE UPLOAD REQUEST")

    # Check if there was a file uploaded
    if not file or file.filename == "":
        logger.warning("❌ Upload rejected: No file provided")
        return templates.TemplateResponse("home.html", {"request": request, "error": "Please upload a file"})
    
    logger.info(f"📁 File received: {file.filename}")
    
    # Check file size (Maximum 30MB)
    file.file.seek(0, os.SEEK_END)
    file_size_bytes = file.file.tell()
    file.file.seek(0)
    max_size_bytes = 30 * 1024 * 1024  # 30MB
    logger.info(f"   Size: {file_size_bytes / 1024:.2f} KB ({file_size_bytes / (1024*1024):.2f} MB)")
    
    if file_size_bytes > max_size_bytes:
        logger.warning(f"❌ Rejected: File size exceeds 30MB limit")
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": "File too large. Maximum allowed size is 30MB."
        })
    logger.info("   ✓ Size validation passed")

    # Validate extension of file to make sure it's only an excel or a CSV file being uploaded
    _, ext = os.path.splitext(file.filename.lower())
    logger.info(f"   Extension: {ext}")
    
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"❌ Rejected: Invalid extension '{ext}'")
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} allowed"
        })
    logger.info("   ✓ Extension validation passed")

    try:
        log_section("⚙️  PROCESSING PIPELINE", "─")
        pipeline_start = time.time()
        
        # Read file
        logger.info("\n[1/4] 📖 Reading file contents...")
        step_start = time.time()
        # ***************************************************************************************************
        # df = read_file(file)
        df = await run_in_threadpool(read_file, file)
        # ***************************************************************************************************

        logger.info(f"      ✓ DataFrame loaded ({time.time() - step_start:.2f}s)")

        # 4. Build file context for the model to have an overview of the file
        logger.info("\n[2/4] 🧠 Building AI context...")
        step_start = time.time()
        # *********************************************************************************************
        # ai_context = make_ai_context(df, file.filename)
        ai_context = await run_in_threadpool(make_ai_context, df, file.filename)
        # *************************************************************************************************
        logger.info(f"      ✓ AI context generated ({time.time() - step_start:.2f}s)")
        logger.info(f"      Context size: {len(ai_context):,} chars")
        
        # Creates a chat session when previous steps have been done
        logger.info("\n[3/4] 🤖 Creating Gemini chat session...")
        logger.info("      Model: gemini-flash-latest | Temp: 0.0")
        step_start = time.time()
        
        # chat_session = client.chats.create(
        #     # model="gemini-2.5-pro",
        #     # model="gemini-2.5-flash",
        #     model ="gemini-flash-latest",
        #     # model="gemini-flash-lite-latest",
        #     # model="gemini-2.5-flash-lite",
        

        #     config=types.GenerateContentConfig(
        #         # system_instruction = SYSTEM_INSTRUCTION + """
        #         # \n\n### CONTEXT OF THE USER'S DATA ###\n""" + ai_context,
        #         system_instruction=f"{SYSTEM_INSTRUCTION}\n\n ###Context of the User's Data\n {ai_context}",

        #         response_mime_type="application/json",
        #         response_schema=list[ModelResponse], 

        #          temperature=0.0
        #     )
        # )

        # *********************************************************************************************
        def create_chat_session():
            return client.chats.create(
                model="gemini-flash-latest",
                config=types.GenerateContentConfig(
                    system_instruction=f"{SYSTEM_INSTRUCTION}\n\n ###Context of the User's Data\n {ai_context}",
                    response_mime_type="application/json",
                    response_schema=list[ModelResponse], 
                    temperature=0.0
                )
            )

        # FIX 3: Run blocking Network call in a thread
        chat_session = await run_in_threadpool(create_chat_session)
        # *********************************************************************************************


        logger.info(f"      ✓ Chat session created ({time.time() - step_start:.2f}s)")
        
        # Save in memory
        logger.info("\n[4/4] 💾 Storing session data...")
        step_start = time.time()
        session_id = str(uuid.uuid4())
        logger.info(f"      Session ID: {session_id[:8]}...")
        
        # Convert file size to KB (rounded to 2 decimals)
        size_kb = file.size / 1024
        if size_kb < 1024:
            file_size = f"{size_kb:.2f} KB"
        else:
            file_size = f"{size_kb/1024:.2f} MB"

        # Getting when the file was uploaded
        current_timestamp = pd.Timestamp.now()

        session_store[session_id] = {
            "df": df,
            "chat_session": chat_session,

            "file_name": file.filename,
            "file_size": file_size,
            "upload_date": current_timestamp.strftime("%Y-%m-%d"),
            "upload_time": current_timestamp.strftime("%I:%M %p"),
            "columns" : list(df.columns),

            "preview_rows": df.head(5).to_dict(orient="records")
        }
        
        logger.info(f"      ✓ Session stored ({time.time() - step_start:.2f}s)")
        
        total_time = time.time() - pipeline_start
        log_section("✅ UPLOAD COMPLETE", "─")
        logger.info(f"""   File: {file.filename}
   Rows: {len(df):,}  |  Columns: {len(df.columns)}
   Active sessions: {len(session_store)}
   ⏱️  Total time: {total_time:.2f}s""")

        

        # Redirect to chat page with session ID
        logger.info(f"\n   → Redirecting to /chat?sid={session_id[:8]}...")
        return RedirectResponse(url=f"/chat?sid={session_id}", status_code=303)

    except Exception as e:
        log_section("❌ UPLOAD FAILED", "═")
        logger.error(f"   Type: {type(e).__name__}")
        logger.error(f"   Message: {str(e)}")
        return templates.TemplateResponse("home.html", {"request": request, "error": str(e)})



