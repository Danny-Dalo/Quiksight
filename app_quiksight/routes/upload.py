
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse
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


# GEMINI FUNCTION CALLING COMMENTED OUT
# return_code_tool = types.Tool(
#     function_declarations=[
#         types.FunctionDeclaration(
#             name="return_code",
#             description="Return only Python code as a string. Do not run it.",
#             parameters=types.Schema(
#                 type=types.Type.OBJECT,
#                 properties={
#                     "code": types.Schema(type=types.Type.STRING)
#                 },
#                 required=["code"]
#             )
#         )
#     ]
# )



# def make_ai_context(df: pd.DataFrame, filename: str, sample_size: int = 5) -> str:

#     context_parts = []

#     # ===== 1. File-level metadata =====
#     context_parts.append(f"📂 Dataset name: {filename}")
#     context_parts.append(f"📐 Shape: {df.shape[0]} rows x {df.shape[1]} columns")

#     # ===== 2. Column summaries =====
#     summaries = []
#     for col in df.columns:
#         dtype = str(df[col].dtype)
#         missing_pct = df[col].isna().mean() * 100
#         unique_vals = df[col].nunique(dropna=True)

#         if pd.api.types.is_numeric_dtype(df[col]):
#             desc = df[col].describe(percentiles=[.25, .5, .75])
#             outliers = ((df[col] < (desc['25%'] - 1.5 * (desc['75%'] - desc['25%']))) |
#                         (df[col] > (desc['75%'] + 1.5 * (desc['75%'] - desc['25%'])))).sum()
#             col_summary = (
#                 f"{col} (numeric) — {dtype}, {unique_vals} unique, "
#                 f"missing: {missing_pct:.1f}%, "
#                 f"min: {desc['min']}, Q1: {desc['25%']}, median: {desc['50%']}, "
#                 f"Q3: {desc['75%']}, max: {desc['max']}, "
#                 f"mean: {desc['mean']:.2f}, std: {desc['std']:.2f}, "
#                 f"outliers: {outliers}"
#             )

#         elif pd.api.types.is_datetime64_any_dtype(df[col]):
#             col_summary = (
#                 f"{col} (datetime) — {dtype}, {unique_vals} unique, "
#                 f"missing: {missing_pct:.1f}%, "
#                 f"range: {df[col].min()} → {df[col].max()}"
#             )

#         else:  # categorical or text
#             top_vals = df[col].value_counts(dropna=True).head(3).to_dict()
#             col_summary = (
#                 f"{col} (categorical/text) — {dtype}, {unique_vals} unique, "
#                 f"missing: {missing_pct:.1f}%, "
#                 f"top values: {top_vals}"
#             )

#         summaries.append(col_summary)

#     context_parts.append("📝 Column summaries:\n" + "\n".join(summaries))

#     # ===== 3. Global dataset stats =====
#     context_parts.append(
#         f"📊 Missing values: {df.isna().sum().sum()} total "
#         f"({df.isna().mean().mean()*100:.1f}% overall)"
#     )
#     context_parts.append(
#         f"🔍 Duplicate rows: {df.duplicated().sum()} "
#         f"({df.duplicated().mean()*100:.1f}% of dataset)"
#     )

#     # ===== 4. Sample rows (head + random sample) =====
#     head_sample = df.head(3).to_dict(orient="records")
#     rand_sample = df.sample(min(sample_size, len(df)), random_state=42).to_dict(orient="records")
#     context_parts.append(f"👀 First rows (preview): {head_sample}")
#     context_parts.append(f"🎲 Random sample rows: {rand_sample}")

#     # ===== 5. Semantic cues =====
#     # A lightweight heuristic “description” the AI can use.
#     numeric_cols = df.select_dtypes(include=np.number).shape[1]
#     cat_cols = df.select_dtypes(exclude=np.number).shape[1]
#     context_parts.append(
#         f"💡 Dataset seems to contain {numeric_cols} numeric features and {cat_cols} categorical/text features."
#     )

#     return "\n\n".join(context_parts)




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
    context_parts = []

    # Existing: File-level metadata
    context_parts.append(f"📂 Dataset name: {filename}")
    context_parts.append(f"📐 Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Existing: Column summaries (unchanged, for brevity)

    # ===== 2. Column summaries =====
    summaries = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_pct = df[col].isna().mean() * 100
        unique_vals = df[col].nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe(percentiles=[.25, .5, .75])
            outliers = ((df[col] < (desc['25%'] - 1.5 * (desc['75%'] - desc['25%']))) |
                        (df[col] > (desc['75%'] + 1.5 * (desc['75%'] - desc['25%'])))).sum()
            col_summary = (
                f"{col} (numeric) — {dtype}, {unique_vals} unique, "
                f"missing: {missing_pct:.1f}%, "
                f"min: {desc['min']}, Q1: {desc['25%']}, median: {desc['50%']}, "
                f"Q3: {desc['75%']}, max: {desc['max']}, "
                f"mean: {desc['mean']:.2f}, std: {desc['std']:.2f}, "
                f"outliers: {outliers}"
            )

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_summary = (
                f"{col} (datetime) — {dtype}, {unique_vals} unique, "
                f"missing: {missing_pct:.1f}%, "
                f"range: {df[col].min()} → {df[col].max()}"
            )

        else:  # categorical or text
            top_vals = df[col].value_counts(dropna=True).head(3).to_dict()
            col_summary = (
                f"{col} (categorical/text) — {dtype}, {unique_vals} unique, "
                f"missing: {missing_pct:.1f}%, "
                f"top values: {top_vals}"
            )

        summaries.append(col_summary)

    context_parts.append("📝 Column summaries:\n" + "\n".join(summaries))

    # ===== 3. Global dataset stats =====
    context_parts.append(
        f"📊 Missing values: {df.isna().sum().sum()} total "
        f"({df.isna().mean().mean()*100:.1f}% overall)"
    )
    context_parts.append(
        f"🔍 Duplicate rows: {df.duplicated().sum()} "
        f"({df.duplicated().mean()*100:.1f}% of dataset)"
    )

    # New: Structural Insights
    context_parts.append("🧩 Structural Insights:")

    # Detect fully empty rows and columns
    empty_rows = df.isnull().all(axis=1).sum()
    empty_cols = df.isnull().all(axis=0).sum()
    context_parts.append(f"  - Fully empty rows: {empty_rows} ({empty_rows / df.shape[0] * 100:.1f}%)")
    context_parts.append(f"  - Fully empty columns: {empty_cols} ({empty_cols / df.shape[1] * 100:.1f}%)")

    # Detect data blocks (contiguous non-empty rows)
    non_empty_mask = ~df.isnull().all(axis=1)
    block_starts = np.where(non_empty_mask & ~non_empty_mask.shift(fill_value=False))[0]
    block_ends = np.where(non_empty_mask & ~non_empty_mask.shift(-1, fill_value=False))[0]
    blocks = []
    for start, end in zip(block_starts, block_ends):
        block_size = end - start + 1
        if block_size > 1:  # Ignore single-row "blocks" (likely notes)
            blocks.append(f"Data block from row {start+1} to {end+1} ({block_size} rows)")
    if blocks:
        context_parts.append("  - Potential multiple tables/sections:\n    " + "\n    ".join(blocks))
    else:
        context_parts.append("  - Appears as a single contiguous table.")

    # Column fill patterns (e.g., columns with data only in certain ranges)
    col_patterns = []
    for col in df.columns:
        non_null_idx = df[col].notnull()
        if non_null_idx.sum() > 0:
            first_data = non_null_idx.idxmax() + 1
            last_data = non_null_idx[::-1].idxmax() + 1
            gaps = (non_null_idx.diff() > 1).sum()  # Rough gap count
            col_patterns.append(f"{col}: Data from row {first_data} to {last_data}, {gaps} gaps")
    if col_patterns:
        context_parts.append("  - Column data ranges:\n    " + "\n    ".join(col_patterns))

    # Existing: Samples, but improved
    # Head + tail + samples from blocks
    head_sample = df.head(3).to_dict(orient="records")
    tail_sample = df.tail(3).to_dict(orient="records")
    rand_samples = []
    for start, end in zip(block_starts, block_ends):
        block_df = df.iloc[start:end+1]
        rand_samples.extend(block_df.sample(min(sample_size // len(blocks) + 1, len(block_df)), random_state=42).to_dict(orient="records"))
    context_parts.append(f"👀 First rows: {head_sample}")
    context_parts.append(f"👀 Last rows: {tail_sample}")
    context_parts.append(f"🎲 Samples from sections: {rand_samples}")

    # Existing: Semantic cues (unchanged)

    return "\n\n".join(context_parts)




SYSTEM_INSTRUCTION = """
# ROLE: JSON Data API
You are a headless data analysis API. Your sole function is to process user requests about a dataset and return a single, valid JSON object. You do not engage in conversation or produce any text outside of the specified JSON structure. Any deviation from this format is a critical failure.

# CRITICAL RULE: JSON OUTPUT ONLY
- Your ENTIRE output, without exception, MUST be a single, valid JSON object.
- DO NOT add any text, explanations, apologies, or markdown like ```json before or after the JSON object.
- The backend system relies exclusively on this JSON format to function.

{
  "response": {
    "code": "Python code runnable as-is that prints a result. The code should be self-contained and not include conversational print statements like 'Here are the results...'. That belongs in the 'text' field.",
    "execution_results": "{{TO_BE_FILLED_BY_BACKEND}}",
    "text": "A short, clear explanation for the user in valid HTML. Use {{EXECUTION_RESULT}} as a placeholder where the output of your 'code' will be injected by the backend."
  }
}

---
# GOAL & BEHAVIOR
You are a senior data assistant helping non-technical users. Your 'text' field should be helpful and human-like; it should blend in seamlessly with the exxecution result, but your overall output must adhere to the JSON structure.

---
# RULES FOR CODE GENERATION
- The dataset is pre-loaded into a pandas DataFrame named `df`.
- Never import libraries or read files.
- Your code MUST handle empty or no-result scenarios gracefully by printing a user-friendly message.
- Before performing any date-based filtering, ensure the relevant column is converted to a datetime format using `pd.to_datetime(df['column_name'], errors='coerce')`.


# RULES FOR OUTPUT FORMATTING
- **NO NESTED COLUMNS:** After any `groupby` or aggregation, you MUST flatten the column headers. Never output a DataFrame with a `MultiIndex`.
  - **Correct Way:** `result = df.groupby('Category').size().reset_index(name='Count')`
  - **Incorrect Way:** `result = df.groupby('Category').agg({'Category': ['count']})`
- **TABLES:** For DataFrames, always use `print(df.to_html(classes='dataframe', index=False))`.
- **SERIES:** A pandas Series (like from `value_counts()`) MUST be converted to a DataFrame with `.to_frame()` before calling `.to_html()`.
- **LISTS:** If the final result is a list of items (e.g., a list of disaster names), convert it to a clean, comma-separated string before printing. Use `print(', '.join(my_list))`. Do not print a raw Python list.
- **SINGLE VALUES:** For single numbers or facts (like shape), print them in a full sentence. Example: `rows, cols = df.shape\\nprint(f"The dataset has {rows} rows and {cols} columns.")`
"""


# Return type because the return type of the function can be a dictionary of file sheets as well
DataFrameOrDict = Union[pd.DataFrame, Dict[str, pd.DataFrame]] 


def read_file(file: UploadFile) -> DataFrameOrDict:
    """
    Reads an uploaded file (CSV or Excel) and returns a DataFrame
    or a dict of DataFrames (if file has multiple sheets).
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
    file.file.seek(0)  # Reset before reading as binary
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
            model="gemini-2.5-pro",
            # model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction = SYSTEM_INSTRUCTION + """
                \n\n### CONTEXT OF THE USER'S DATA ###\n""" + ai_context,
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



