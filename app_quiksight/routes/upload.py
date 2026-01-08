
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import os
import pandas as pd
import io, csv
import numpy as np


from typing import Union, Dict

from api_training2.config import GEMINI_API_KEY
import uuid

from google import genai
from google.genai import types

router = APIRouter()
templates = Jinja2Templates(directory="app_quiksight/templates")

ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

# In-memory storage (will replace with DB/Redis)
session_store = {}

client = genai.Client(api_key=GEMINI_API_KEY)


# ========== Need to review these functions ============
def make_ai_context(df: Union[pd.DataFrame, Dict[str, pd.DataFrame]], filename: str, sample_size: int = 5) -> str:
    if isinstance(df, pd.DataFrame):
        return _build_context_for_df(df, filename, sample_size)
    else:
        # Multi-sheet: Build context for each
        contexts = []
        for sheet_name, df in df.items():
            contexts.append(f"📑 Sheet: {sheet_name}\n" + _build_context_for_df(df, filename, sample_size))
        return "\n\n---\n\n".join(contexts)


def _build_context_for_df(df: pd.DataFrame, filename: str, sample_size: int) -> str:
    """Build a token-efficient context summary for the AI."""
    context_parts = []
    num_cols = len(df.columns)
    num_rows = len(df)
    
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
    
    # Check for multiple data blocks (only report if fragmented)
    non_empty_mask = ~df.isnull().all(axis=1)
    block_starts = np.where(non_empty_mask & ~non_empty_mask.shift(fill_value=False))[0]
    if len(block_starts) > 1:
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

    filename = file.filename.lower()

    # ---- CSV Handling ----
    if filename.endswith(".csv"):
        for encoding in ["utf-8", "latin1", "iso-8859-1", "cp1252"]:
            try:
                file.file.seek(0)
                # when file is read, pointer reads it and goes to the end (like when you read a book)
                # If a previous read failed, the cursor is already at the end so it would be seen as empty if you try again
                # .seek() resets it back to the beginning (going back to the beginning of the book) to read again
                return pd.read_csv(file.file,
                                   encoding=encoding,
                                   engine="python",
                                   quotechar='"',
                                   quoting=csv.QUOTE_MINIMAL,
                                   skip_blank_lines=True,
                                   )
            except UnicodeDecodeError:
                continue
        raise ValueError("Unable to decode CSV with supported encodings.")

    # ---- Excel Handling ----
    file.file.seek(0)  # takes pointer to beginning of the file
    raw = file.file.read()

    try:
        xls = pd.ExcelFile(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")

    sheets = xls.sheet_names
    if not sheets:
        raise ValueError("No sheets found in Excel file.")

    if len(sheets) != 1:
        raise ValueError("Only Excel files with a single sheet are supported at this time.")

    return pd.read_excel(xls, sheet_name=sheets[0])





from pydantic import BaseModel
class ModelResponse(BaseModel):
    text_explanation: str
    code_generated: str
    should_execute : bool


# Uploaded files are validated and submitted to the chat endpoint for the AI model to use
@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):

    # Check if there was a file uploaded
    if not file or file.filename == "":
        return templates.TemplateResponse("home.html", {"request": request, "error": "Please upload a file"})
    
    # Check file size (Maximum 30MB)
    file.file.seek(0, os.SEEK_END)
    file_size_bytes = file.file.tell()
    file.file.seek(0)
    max_size_bytes = 30 * 1024 * 1024  # 30MB
    if file_size_bytes > max_size_bytes:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": "File too large. Maximum allowed size is 30MB."
        })

    # Validate extension of file to make sure it's only an excel or a CSV file being uploaded
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} allowed"
        })
    

    try:
        # Read file
        df = read_file(file)

        # 4. Build file context for the model to have an overview of the file
        ai_context = make_ai_context(df, file.filename)
        
        # Creates a chat  session when previous steps have been done
        chat_session = client.chats.create(
            # model="gemini-2.5-pro",
            # model="gemini-2.5-flash",
            model ="gemini-flash-latest",
            # model="gemini-flash-lite-latest",
            # model="gemini-2.5-flash-lite",
        

            config=types.GenerateContentConfig(
                # system_instruction = SYSTEM_INSTRUCTION + """
                # \n\n### CONTEXT OF THE USER'S DATA ###\n""" + ai_context,
                system_instruction=f"{SYSTEM_INSTRUCTION}\n\n ###Context of the User's Data\n {ai_context}",

                response_mime_type="application/json",
                response_schema=list[ModelResponse], 

                 temperature=0.0
            )
        )

        
       
        
        # Save in memory
        session_id = str(uuid.uuid4())
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

        

        # Redirect to chat page with session ID
        return RedirectResponse(url=f"/chat?sid={session_id}", status_code=303)

    except Exception as e:
        return templates.TemplateResponse("home.html", {"request": request, "error": str(e)})



