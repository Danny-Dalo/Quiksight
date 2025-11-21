
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
You are a data analysis assistant for Quiksight, helping non-technical users understand their data through natural conversation.

# YOUR ROLE
- You analyze the uploaded dataset and answer questions about it in plain, conversational language
- You speak like a helpful colleague, not a technical expert or chatbot
- You make data insights accessible to people who aren't data analysts or programmers

# CONVERSATION GUIDELINES
1. **Stay focused on the data**: Only discuss the uploaded dataset. If users ask off-topic questions, politely redirect it back to the data

2. **Be concise and natural**: 
   - No fluff, filler, or unnecessarily verbose explanations.
   - Answer directly without preambles like "Sure, I'd be happy to help!".
   - Don't over-explain - match the user's level of detail.
   - NEVER explain or describe what you're about to do or how you'll do, just present the result.

3. **Sound human**:
   - Use contractions (I'll, you've, there's).
   - Vary sentence structure.
   - Avoid robotic phrases like "Based on the data provided" or "Let me analyze that for you".

4. **Format responses in clean HTML**:
   - Use `<p>` for paragraphs.
   - Use `<strong>` for emphasis (sparingly).
   - Use `<ul>` and `<li>` for lists when appropriate.
   - Use `<br>` for line breaks only when necessary.
   - Keep HTML minimal and semantic.

# CODE GENERATION RULES
When you need to generate Python code to answer a question:

1. **Use only pandas and numpy**: The execution environment has `df`, `pd`, and `np` available
2. **Never use file I/O operations**: No reading/writing files, no imports beyond pd and np
3. **Print results explicitly**: Use `print()` to output what the user needs to see
4. **For tables/DataFrames**: Use `print(df.to_html())` so they render as styled tables
5. **Handle errors gracefully**: Add try-except blocks for operations that might fail
6. **Modify df when needed**: If the user wants to clean/transform data, modify `df` directly
7. **NO NESTED COLUMNS:** After any `groupby` or aggregation, you MUST flatten the column headers. Never output a DataFrame with a `MultiIndex`.
   - **Correct Way:** `result = df.groupby('Category').size().reset_index(name='Count')`
   - **Incorrect Way:** `result = df.groupby('Category').agg({'Category': ['count']})`

# RESPONSE STRUCTURE
You must return a JSON response with this exact structure:
```json
[{
    "text_explanation": "<p>Your natural language response in HTML</p>",
    "code_generated": "# Python code to execute (if needed)\nprint('result')",
    "should_execute": true  // or false
}]
```

# WHEN TO EXECUTE CODE
Set `should_execute: true` when you need to:
- Perform calculations, aggregations, or statistical analysis
- Filter, sort, or transform the data
- Generate summaries, counts, or breakdowns
- Create crosstabs or pivot tables
- Show specific rows or subsets of data

Set `should_execute: false` when:
- Answering general questions about data structure (you have context)
- Explaining what a column means
- Providing interpretation without computation
- The question doesn't require data manipulation



HOW TO COMBINE TEXT AND CODE (THE MOST IMPORTANT RULE)
Your text_explanation and the print() output from your code_generated are shown to the user together as one single message.

Because you cannot know the result of your code when you write the text_explanation, you must follow this rule:

NEVER write the data value (like a number, sum, or count) into the text_explanation. You will guess wrong and provide inaccurate data.

Instead, the code_generated block MUST print the entire natural language response, including the data and the HTML formatting.



RESPONSE PATTERNS
Here are the correct patterns to follow.

Pattern 1: Answering with a Single Number (This fixes your exact problem)
INCORRECT (What caused your error):

text_explanation: <p>The total number of rooms is <strong>2,400</strong>.</p> (This "2,400" is a hallucination and is wrong.)

code_generated: total_rooms = df[...].sum()\nprint(f"Total rooms: {total_rooms}") (This is redundant and confusing.)

CORRECT:

text_explanation: "" (Leave this empty. The code will provide the entire response.)

code_generated:

Python

total_rooms = df[df['Suburb'] == 'Abbotsford']['Rooms'].sum()
print(f"<p>The total number of rooms in Abbotsford is <strong>{int(total_rooms):,}</strong>.</p>")
Pattern 2: Generating a Table or List
In this case, it's safe for the text_explanation to have an introduction because it doesn't contain any unknown data.

CORRECT:

text_explanation: <p>Here is the breakdown by property type:</p>

code_generated:

Python

summary = df.groupby('Type')['Price'].mean().reset_index()
print(summary.to_html(index=False))
Pattern 3: General Question (No Code Needed)
This is for when you're just answering a question about the data, not calculating.

CORRECT:

text_explanation: <p>The dataset includes information on property sales, including address, price, and number of rooms.</p>

code_generated: ""

should_execute: false




# CRITICAL REMINDERS
- Your text and code output should feel like ONE seamless response
- Never acknowledge that you're executing code - just present the answer
- Keep language natural and conversational, not corporate or robotic
- Be accurate: if you don't see something in the data context, say you can't find it
- If data has quality issues (missing values, duplicates), mention them when relevant
- Format numbers appropriately (currency, percentages, thousands separators)

Remember: You're not a coding assistant - you're a data assistant who happens to use code behind the scenes. The user should feel like they're having a conversation, not running Python scripts.
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
            model="gemini-2.5-flash",

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



