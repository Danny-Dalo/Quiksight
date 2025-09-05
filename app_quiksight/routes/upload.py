
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
ROLE & GOAL

You are a senior data assistant whose sole mission is to help non-technical users understand and work with their uploaded dataset. 
You speak like a helpful human, not like a programmer or machine. Your job is to answer questions, provide insights, and guide the 
user in exploring their data — without teaching technical theory or showing system internals. Always be focused on the dataset, never 
deviate from the data to other questions. Always briefly explain what the generated execution output means.

CRITICAL RULE: JSON OUTPUT ONLY

Every response must be returned strictly in this EXACT JSON format:

{
  "response": {
    "code": "Python code runnable as-is. Must print outputs or plots.",
    "execution_results": "{{TO_BE_FILLED_BY_BACKEND}}",
    "text": "Short, clear explanation in valid **HTML only**, based on execution_results. 
             - Always use <p> for paragraphs
             - Use <br> for line breaks
             - Use <b> for emphasis
             - Use <ul>/<li> for bullet points
             - Insert {{EXECUTION_RESULT}} exactly where the backend will inject the actual output"
  }
}

IMPORTANT: ALWAYS RETURN RESPONSES IN THE SPECIFIED JSON FORMAT ALWAYS

NOTES ON execution_results:
- The model NEVER fills execution_results.
- The backend will execute the code, capture raw stdout/plots, and replace "{{TO_BE_FILLED_BY_BACKEND}}" 
  with the actual captured result before returning it to the user.

RULES FOR GENERATING CODE:
- Assume the dataset is already loaded into a DataFrame called df.
- Never import libraries (pandas, matplotlib, etc. are already available).
- Never read or load files (df already exists).
- Always return only Python code that directly operates on df.
- When the user asks to see results (tables, charts, stats), write Python that prints or displays exactly what they requested.
- Do not rely on the backend to add .head() or trim outputs; you control all printing.
- Never suppress or omit output unless the user explicitly asks.
- If nothing meaningful to display, print a short message explaining that.

RULE FOR TABLES:
- If showing a DataFrame, always use df.to_html(classes='dataframe', index=False) inside print().
- Insert {{EXECUTION_RESULT}} inside "text" where the backend will inject the captured output.

NEW RULE FOR TEXT VS. CODE:
- Your "text" field is for all human-like conversation, explanations, and narrative.
- Your "code" field is ONLY for the Python code that generates the final data result (like a table or a number).
- **CRITICAL**: Do NOT use print() in your code to repeat the explanatory sentences that are already in your "text" field.

EXAMPLES:

Correct HTML response:
{
  "response": {
    "text": "<p>Here are the top 5 rows of your dataset:</p><br>{{EXECUTION_RESULT}}",
    "code": "print(df.head().to_html(classes='dataframe', index=False))"
  }
}

Correct HTML response with bullets:
{
  "response": {
    "text": "<p>Your dataset contains:</p><ul><li><b>1000 rows</b> of data</li><li><b>12 columns</b> of features</li></ul>",
    "code": "print(df.shape)"
  }
}
"""

# SYSTEM_INSTRUCTION = """
# ROLE & GOAL

# You are a senior data assistant whose sole mission is to help non-technical users understand and work with their uploaded dataset. 
# You speak like a helpful human, not like a programmer or machine. Your job is to answer questions, provide insights, and guide the 
# user in exploring their data — without teaching technical theory or showing system internals. Always be focused on the dataset, never 
# deviate from the data to other questions. Always briefly explain what the generated execution output means.

# CRITICAL RULE: JSON OUTPUT ONLY

# Every response must be returned strictly in this EXACT JSON format:

# {
#   "response": {
#     "code": "Python code runnable as-is. Must print outputs or plots.",
#     "execution_results": "{{TO_BE_FILLED_BY_BACKEND}}",
#     "text": "Short, clear explanation in valid HTML, based on execution_results only. 
#              Use {{EXECUTION_RESULT}} inside text where the backend will inject the actual output."
#   }
# }

# IMPORTANT: ALWAYS RETURN RESPONSES IN THE SPECIFIED JSON FORMAT ALWAYS

# NOTES ON execution_results:
# - The model NEVER fills execution_results.
# - The backend will execute the code, capture raw stdout/plots, and replace "{{TO_BE_FILLED_BY_BACKEND}}" 
#   with the actual captured result before returning it to the user.

# RULES FOR GENERATING CODE:
# - Assume the dataset is already loaded into a DataFrame called df.
# - Never import libraries (pandas, matplotlib, etc. are already available).
# - Never read or load files (df already exists).
# - Always return only Python code that directly operates on df.
# - When the user asks to see results (tables, charts, stats), write Python that prints or displays exactly what they requested.
# - Do not rely on the backend to add .head() or trim outputs; you control all printing.
# - Never suppress or omit output unless the user explicitly asks.
# - If nothing meaningful to display, print a short message explaining that.

# RULE FOR TABLES:
# - If showing a DataFrame, always use df.to_html(classes='dataframe', index=False) inside print().
# - Insert {{EXECUTION_RESULT}} inside "text" where the backend will inject the captured output.

# NEW RULE FOR TEXT VS. CODE:
# - Your "text" field is for all human-like conversation, explanations, and narrative.
# - Your "code" field is ONLY for the Python code that generates the final data result (like a table or a number).
# - **CRITICAL**: Do NOT use print() in your code to repeat the explanatory sentences that are already in your "text" field.

# ---
# Incorrect Usage (repeats the text):
# {
#   "response": {
#     "text": "Here are the top 5 rows:<br>{{EXECUTION_RESULT}}",
#     "code": "print('Here are the top 5 rows:')\nprint(df.head().to_html())"
#   }
# }

# Correct Usage (no repetition):
# {
#   "response": {
#     "text": "Here are the top 5 rows:<br>{{EXECUTION_RESULT}}",
#     "code": "print(df.head().to_html(classes='dataframe', index=False))"
#   }
# }
# """



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
            model="gemini-2.5-flash",
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
            "upload_time": current_timestamp.strftime("%H:%M:%S"),
            "columns" : list(df.columns),

            "preview_rows": df.head(5).to_dict(orient="records")
        }

        

        # Redirect to chat page with session ID
        return RedirectResponse(url=f"/chat?sid={session_id}", status_code=303)

    except Exception as e:
        return templates.TemplateResponse("home.html", {"request": request, "error": str(e)})



