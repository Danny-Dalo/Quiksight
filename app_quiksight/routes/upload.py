from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import time
import shutil
import uuid
import logging
from datetime import datetime
from typing import Dict, Any

from pydantic import BaseModel

# Configure logging
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

# Session store variable is expected to be a dictionary of {"key" : Any}
session_store: Dict[str, Any] = {}

class ModelResponse(BaseModel):
    text_explanation: str
    code_generated: str
    should_execute : bool

@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    log_section("📤 NEW FILE UPLOAD REQUEST")

    # 1. Validation
    if not file or file.filename == "":
        logger.warning("❌ Upload rejected: No file provided")
        return templates.TemplateResponse("home.html", {"request": request, "error": "Please upload a file"})
    
    # Size check (30MB)
    file.file.seek(0, os.SEEK_END)
    file_size_bytes = file.file.tell()
    file.file.seek(0)
    if file_size_bytes > 30 * 1024 * 1024:
        logger.warning(f"❌ Rejected: File too large")
        return templates.TemplateResponse("home.html", {"request": request, "error": "File too large (Max 30MB)"})

    # Ext check
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"❌ Rejected: Invalid extension {ext}")
        return templates.TemplateResponse("home.html", {"request": request, "error": "Invalid file type"})

    # 2. Save File
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True) # Ensure dir exists
    
    session_id = str(uuid.uuid4())
    file_path = os.path.join(upload_dir, f"{session_id}{ext}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"   ✓ File saved to {file_path}")
    except Exception as e:
        logger.error(f"   ❌ Failed to save file: {e}")
        return templates.TemplateResponse("home.html", {"request": request, "error": "Failed to save file"})

    # 3. Create Pending Session
    now = datetime.now()
    
    # Calculate size string
    size_kb = file_size_bytes / 1024
    file_size_str = f"{size_kb:.2f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB"

    session_store[session_id] = {
        "status": "pending",
        "file_path": os.path.abspath(file_path),
        "file_name": file.filename,
        "file_size": file_size_str,
        "upload_date": now.strftime("%Y-%m-%d"),
        "upload_time": now.strftime("%I:%M %p"),
        "df": None, 
        "chat_session": None,
        "columns": [],
        "preview_rows": [],
        "num_rows": 0,
        "num_columns": 0
    }
    
    logger.info(f"   ✓ Session {session_id[:8]} created (Pending)")
    logger.info(f"   → Redirecting to /chat?sid={session_id}")
    
    return RedirectResponse(url=f"/chat?sid={session_id}", status_code=303)
