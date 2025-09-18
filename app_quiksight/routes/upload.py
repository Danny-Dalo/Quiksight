
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import os
import pandas as pd
import io, csv
import numpy as np

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



def make_ai_context(df: pd.DataFrame, filename: str, sample_size: int = 5) -> str:
    context_parts = []

    # ===== 1. File-level metadata =====
    context_parts.append(f"📂 Dataset name: {filename}")
    context_parts.append(f"📐 Shape: {df.shape[0]} rows x {df.shape[1]} columns")

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

    # ===== 4. Sample rows (head + random sample) =====
    head_sample = df.head(3).to_dict(orient="records")
    rand_sample = df.sample(min(sample_size, len(df)), random_state=42).to_dict(orient="records")
    context_parts.append(f"👀 First rows (preview): {head_sample}")
    context_parts.append(f"🎲 Random sample rows: {rand_sample}")

    # ===== 5. Semantic cues =====
    # A lightweight heuristic “description” the AI can use.
    numeric_cols = df.select_dtypes(include=np.number).shape[1]
    cat_cols = df.select_dtypes(exclude=np.number).shape[1]
    context_parts.append(
        f"💡 Dataset seems to contain {numeric_cols} numeric features and {cat_cols} categorical/text features."
    )

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
- Never import libraries (like pandas, numpy) or read files. They are already available.
- Your code MUST handle empty or no-result scenarios gracefully by printing a user-friendly message (e.g., `print("No results found for your query.")`).
- **For DataFrames:** Always use `print(df.to_html(classes='dataframe', index=False))` to output tables.
- **For pandas Series:** A Series (like from `value_counts()`) MUST be converted to a DataFrame before outputting. Use `print(my_series.to_frame().to_html(classes='dataframe'))`.
- **For Lists/Tuples:** Format them for readability. For example: `rows, cols = df.shape\\nprint(f"The dataset has {rows} rows and {cols} columns.")` instead of just printing the tuple.
"""

# After a file with valid extension has been uploaded, this function reads and loads the file (excel/csv)
def read_file(file: UploadFile) -> pd.DataFrame:
    raw = file.file.read()
    if file.filename.lower().endswith(".csv"):
        for encoding in ["utf-8", "latin1", "iso-8859-1", "cp1252"]:
            try:
                return pd.read_csv(io.StringIO(raw.decode(encoding)),
                                   engine="python", quotechar='"',
                                   quoting=csv.QUOTE_MINIMAL,
                                   skip_blank_lines=True)
            except UnicodeDecodeError:
                continue
        raise ValueError("Unable to decode CSV with supported encodings.")
    else:
        return pd.read_excel(io.BytesIO(raw))







# Uploaded files are validated and submitted to the chat endpoint for the AI model to use
@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    # 1. No file
    if not file or file.filename == "":
        return templates.TemplateResponse("home.html", {"request": request, "error": "Please upload a file"})

    # 2. Validate extension
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} allowed"
        })

    try:
        # 3. Read file
        df = read_file(file)

        # 4. Build AI context
        ai_context = make_ai_context(df, file.filename)

        # Creating a chat  session
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
            # "context": ai_context,

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



