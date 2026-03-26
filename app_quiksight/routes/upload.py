from fastapi import APIRouter, File, UploadFile, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app_quiksight.storage.auth_deps import get_optional_user
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

# logging config
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

# SYSTEM_INSTRUCTION = """
# You are an experienced data analyst helping a user explore their dataset through conversation.

# Your responses MUST be valid, clean HTML only.
# Do NOT use Markdown.
# Do NOT use backticks.
# Do NOT use plain text formatting symbols like ** or - for bullets.
# Only return properly structured HTML.

# GENERAL BEHAVIOR
# - Respond naturally, like a smart colleague.
# - Do not perform a full dataset summary unless the user explicitly asks for one.
# - If the user greets you or makes small talk, respond briefly and naturally.
# - Only reference the dataset when relevant to the user's question.
# - Do not overwhelm the user with unnecessary analysis.

# WHEN ANSWERING DATA QUESTIONS
# - Base responses strictly on the dataset context provided.
# - Do not invent numbers.
# - If something cannot be determined from the context, say so clearly.
# - Provide insight when appropriate, but keep it proportional to the question.

# HTML FORMATTING RULES
# - Always wrap content in semantic HTML elements.
# - Use <p> for paragraphs.
# - Use <h2> or <h3> for section headers when useful.
# - Use <ul>, <ol> and <li> when listing out items or points.
# - use <br/> to create spaces between text sections and to make responses not to be clustered.
# - Use <hr/> when dividing section blocks.
# - Use <strong> for important numbers or findings.
# - Use <table>, <thead>, <tbody>, <th>, <tr>, <td> for any data that has multiple columns per item.
#   Specifically, ALWAYS use a table when:
#     • Showing rankings, top/bottom lists, or leaderboards
#     • Comparing multiple entities across multiple attributes (e.g. name + amount + quantity)
#     • Displaying grouped or aggregated results (e.g. per-region totals)
#     • The data has 2 or more fields per row
#   Use <ol>/<ul> ONLY for simple single-value lists or short bullet points (e.g. "key takeaways").
# - Keep structure clean and modern.
# - Do NOT overuse headers or lists or an of the elements.
# - Avoid excessive nesting.
# - No inline styles.
# - No CSS.
# - No Markdown.

# TONE
# - Clear, direct, conversational.
# - No forced enthusiasm.
# - No filler phrases like "Great question!"
# - Avoid sounding like a formal research report.
# - Keep paragraphs reasonably short.

# IMPORTANT
# - Do not proactively summarize the entire dataset unless asked.
# - Let the conversation unfold naturally.
# - Output must be valid HTML and nothing else.
# """

SYSTEM_INSTRUCTION = """
You are a knowledgeable analyst helping a user explore and understand their data through conversation.

Your responses MUST be valid, clean HTML only.
Do NOT use Markdown.
Do NOT use backticks.
Do NOT use plain text formatting symbols like ** or - for bullets.
Only return properly structured HTML.

GENERAL BEHAVIOR
- Respond naturally, like a smart colleague who already knows the data well.
- Do not volunteer a full analysis unless the user explicitly asks for one.
- If the user greets you or makes small talk, respond briefly and naturally.
- Only bring up data when it is directly relevant to what the user is asking.
- Do not overwhelm the user with unsolicited detail.

WHEN ANSWERING DATA QUESTIONS
- Answer based only on what you know from the data available to you.
- Do not invent or estimate numbers.
- Never reference how you access information, what format it is in, or how it was provided to you.
- If something cannot be determined, say so plainly without explaining why.
- Provide insight where appropriate, but keep it proportional to the question.

HTML FORMATTING RULES
- Always wrap content in semantic HTML elements.
- Use <p> for paragraphs.
- Use <h2> or <h3> for section headers when useful.
- Use <ul>, <ol> and <li> when listing out items or points.
- Use <br/> to create breathing room between text sections.
- Use <hr/> when dividing distinct sections.
- Use <strong> for important numbers or findings.
- Use <table>, <thead>, <tbody>, <th>, <tr>, <td> for any data with multiple columns per item.
  Specifically, ALWAYS use a table when:
    • Showing rankings, top/bottom lists, or leaderboards
    • Comparing multiple entities across multiple attributes
    • Displaying grouped or aggregated results
    • The data has 2 or more fields per row
  Use <ol>/<ul> ONLY for simple single-value lists or short bullet points.
- Keep structure clean and modern.
- Do NOT overuse headers, lists, or any structural elements.
- Avoid excessive nesting.
- No inline styles.
- No CSS.
- No Markdown.

TONE
- Clear, direct, conversational.
- No forced enthusiasm.
- No filler phrases like "Great question!"
- Avoid sounding like a formal research report.
- Keep paragraphs short.

IMPORTANT
- Never expose implementation details, technical context, or how information was supplied to you.
- Do not proactively summarize everything unless the user asks.
- Let the conversation unfold naturally.
- Output must be valid HTML and nothing else.
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

        is_bool = pd.api.types.is_bool_dtype(df[col])
        is_numeric = pd.api.types.is_numeric_dtype(df[col]) and not is_bool

        if is_numeric:
            desc = df[col].describe(percentiles=[.25, .5, .75])
            min_val = desc.get('min', float('nan'))
            max_val = desc.get('max', float('nan'))
            mean_val = desc.get('mean', float('nan'))
            
            min_str = f"{min_val:.4g}" if pd.notna(min_val) else "NaN"
            max_str = f"{max_val:.4g}" if pd.notna(max_val) else "NaN"
            mean_str = f"{mean_val:.4g}" if pd.notna(mean_val) else "NaN"
            
            col_summary = (
                f"{col} (num) — {unique_vals} unique, "
                f"range: {min_str}–{max_str}, "
                f"mean: {mean_str}, missing: {missing_pct:.0f}%"
            )
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_summary = (
                f"{col} (date) — {unique_vals} unique, "
                f"range: {df[col].min()} → {df[col].max()}, missing: {missing_pct:.0f}%"
            )
        else:  # categorical, boolean, or text
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
        raise ValueError("We couldn't read your CSV file. Please make sure it's a valid CSV with standard encoding.")

    # Excel Handling
    file.file.seek(0)
    raw = file.file.read()
    try:
        xls = pd.ExcelFile(io.BytesIO(raw))
    except Exception as e:
        logger.error(f"Excel read failed: {e}")
        raise ValueError("We couldn't read your Excel file. Please make sure it's a valid .xlsx or .xls file.")

    sheets = xls.sheet_names
    if not sheets: raise ValueError("Your Excel file doesn't appear to contain any data sheets.")
    
    # If multiple sheets, return dictionary of dataframes
    if len(sheets) == 1:
        return pd.read_excel(xls, sheet_name=sheets[0])
    else:
        return pd.read_excel(xls, sheet_name=None)  # Returns Dict[str, DataFrame]



# ==============================================================================
#                                   UPLOAD ROUTE
# ==============================================================================

@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, files: list[UploadFile] = File(...), user = Depends(get_optional_user)):
    if not user:
        return RedirectResponse(url="/login", status_code=303)
        
    log_section("       ==NEW FILE UPLOAD REQUEST==       ")

    """Check if files are actually uploaded when request is made"""
    if not files or all(f.filename == "" for f in files):
        return templates.TemplateResponse("home.html", {"request": request, "error": "Please upload a file"})
    
    """ Checks if the uploaded files extensions are allowed """
    for file in files:
        if file.filename:
            _, ext = os.path.splitext(file.filename.lower())
            if ext not in ALLOWED_EXTENSIONS:
                return templates.TemplateResponse("home.html", {"request": request, "error": f"Invalid file type for {file.filename}"})

    try:
        log_section("    PROCESSING PIPELINE    ", "─")
        pipeline_start = time.time()    # tracking how fast it takes to complete processes
        
        all_dfs = {}
        total_size_bytes = 0
        file_names = []
        file_exts = set()
        
        for file in files:
            if not file.filename: continue
            
            logger.info(f"\n Step [1/3]  Reading file: {file.filename}...")
            df_or_dict = read_file(file)
            
            file_names.append(file.filename)
            total_size_bytes += file.size
            _, ext = os.path.splitext(file.filename.lower())
            file_exts.add(ext)

            if isinstance(df_or_dict, pd.DataFrame):
                all_dfs[file.filename] = df_or_dict
            else:
                for sheet_name, sheet_df in df_or_dict.items():
                    all_dfs[f"{file.filename} - {sheet_name}"] = sheet_df

        """Summarize data and generate data context for the AI model"""
        logger.info("\n Step [2/3]  Generating AI Context...")
        # Make a combined context
        contexts = []
        for name, df in all_dfs.items():
            contexts.append(f"📁 Dataset: {name}\n" + make_ai_context(df, name))
        data_context = "\n\n---\n\n".join(contexts)

        """Create a session ID for each uploaded file for reference"""
        logger.info("\n Step [3/3] Creating Session...")
        session_id = str(uuid.uuid4())
        
        """Calculating file size(in KB and MB)"""
        size_kb = total_size_bytes / 1024
        file_size = f"{size_kb:.2f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB"
        current_timestamp = datetime.datetime.now()

        # Sanitize DFs and Save Parquet
        dataframe_paths = {}
        preview_data = {}
        columns_dict = {}
        total_rows = 0
        
        # Save Parquet for each dataframe
        for name, df in all_dfs.items():
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
            
            # Safe filename for parquet
            safe_name = "".join(c if c.isalnum() else "_" for c in name)
            df_path = f"{DATA_DIR}/{session_id}_{safe_name}.parquet"
            df.to_parquet(df_path, engine="pyarrow")
            
            dataframe_paths[name] = df_path
            preview_data[name] = df.head(5).astype(str).to_dict(orient="records")
            columns_dict[name] = list(df.columns)
            total_rows += len(df)

        session_data = {
            "file_name": ", ".join(file_names),
            "file_size": file_size,
            "file_extension": ", ".join(file_exts),
            "upload_date": str(current_timestamp.strftime("%Y-%m-%d")),
            "upload_time": str(current_timestamp.strftime("%I:%M %p")),
            "columns": json.dumps(columns_dict),
            "num_rows": total_rows,
            "preview_rows": json.dumps(preview_data),
            "dataframe_paths": json.dumps(dataframe_paths),
            "data_context": data_context,
            "user_id": user.id,
        }

        """ Redis stores everything necessary to resume the chat session later as long as it has not expired"""
        redis_key = f"session:{session_id}"
        redis_client.hset(redis_key, mapping=session_data)
        redis_client.expire(redis_key, 3600 * 24 * 7)  # 7 day expiry for session persistence

        # Track this session under the user's session list
        user_sessions_key = f"user_sessions:{user.id}"
        redis_client.sadd(user_sessions_key, session_id)
        redis_client.expire(user_sessions_key, 3600 * 24 * 30)  # 30 day expiry for user session index

        total_time = time.time() - pipeline_start   # How long the whole upload process took
        log_section("               UPLOAD COMPLETE     ", "─")
        logger.info(f"   Session: {session_id[:8]}...{session_id[-4:]} | Time: {total_time:.2f}s")
        
        return RedirectResponse(url=f"/chat?sid={session_id}", status_code=303)

    except ValueError as e:
        # ValueError for failed upload
        log_section("                UPLOAD FAILED      ", "═")
        logger.error(f"   Validation Error: {str(e)}")
        return templates.TemplateResponse("home.html", {"request": request, "error": str(e)})
    except Exception as e:
        # Unexpected errors so it doesn't leak technical details on frontend
        log_section("                UPLOAD FAILED      ", "═")
        logger.error(f"   Unexpected Error: {str(e)}")
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": "Something went wrong while processing your file. Please try again or use a different file."
        })
