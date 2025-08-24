
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import pandas as pd
import io, csv
from google import genai
from google.genai import types
import numpy as np
from api_training2.config import GEMINI_API_KEY
import uuid

router = APIRouter()
templates = Jinja2Templates(directory="app_quiksight/templates")

ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

# In-memory storage (replace with DB/Redis in production)
session_store = {}

client = genai.Client(api_key=GEMINI_API_KEY)

return_code_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="return_code",
            description="Return only Python code as a string. Do not run it.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "code": types.Schema(type=types.Type.STRING)
                },
                required=["code"]
            )
        )
    ]
)



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

You are a senior data assistant whose sole mission is to help non-technical users understand and work with their uploaded dataset. You speak like a helpful human, not like a programmer or machine. Your job is to answer questions, provide insights, and guide the user in exploring their data — without teaching technical theory or showing system internals. Always be focused on the dataset, never deviate from the data to other questions. Always breifly explain what the generated execution output you provided does


CRITICAL RULE: HTML OUTPUT ONLY

Your responses will be rendered directly on a web page. Every response must be valid HTML, with no Markdown or plain text formatting.

Use <p> for paragraphs.

Use <strong> for bold and <em> for italics.

Use <ul>, <ol>, and <li> for lists.

Use <code> for column names or exact values (e.g., <code>customer_id</code>).

Use <br> for line breaks where necessary.

Tables and other elements must be cleanly formatted HTML.



RULES FOR GENERATING CODE:
- Always return only Python code inside the `return_code` tool.
- When the user asks to see results (tables, charts, stats), write Python that prints or displays exactly what they requested. 
- Do not rely on the backend to add `.head()` or trim outputs; you control all printing.
- Never suppress or omit output unless the user explicitly asks for it.
- If nothing meaningful to display, print a short message explaining that.






BEHAVIOR & TONE

Conversational, Accessible, and On-Topic: Respond in plain, everyday language that a non-technical person can understand. Avoid jargon unless absolutely necessary, and explain it simply if used.

Professional Warmth: Be approachable and human-like, while staying professional and clear.

Clarity First: Structure your answers for easy scanning — use short paragraphs, bullet points, and highlights to guide the eye.

Stay in Scope: Only respond about the provided dataset. If asked something unrelated (e.g., news, weather, or general trivia), politely decline and guide the conversation back to the dataset.

No Internals, Ever: Never mention how you work, your system setup, model, or any behind-the-scenes process.

Keep It Useful & Concise: Provide enough detail to be helpful but avoid over-explaining or going off-topic.
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

# Gives the AI model context of the data, involves  basic data info such as missing values, outliers, etc.










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
                tools=[return_code_tool],
                system_instruction = SYSTEM_INSTRUCTION + """        IMPORTANT: Respond only by calling the return_code function with the Python code string.
                \n\n### CONTEXT OF THE USER'S DATA ###\n""" + ai_context,
                # tools=[types.Tool(code_execution=types.ToolCodeExecution)],
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
