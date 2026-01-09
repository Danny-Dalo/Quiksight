"""
Upload route - ZERO-CPU upload endpoint.
Stores raw JSON data. DataFrame is created LAZILY when chat needs it.
This achieves <100ms upload time on Render free tier (0.1 CPU).
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
from datetime import datetime

router = APIRouter()

# In-memory session storage - stores RAW data, not DataFrames
session_store: Dict[str, Dict[str, Any]] = {}

# Constants
MAX_ROWS = 100_000
MAX_COLUMNS = 500


class UploadPayload(BaseModel):
    """Schema for client-side parsed data."""
    filename: str
    file_size: int
    columns: List[str]
    rows: List[Dict[str, Any]]


@router.post("/upload")
async def upload_parsed_data(payload: UploadPayload):
    """
    Ultra-fast upload - just validate and store raw JSON.
    NO pandas, NO DataFrame creation here.
    DataFrame is created lazily in chat.py when first message is sent.
    """
    # Quick validations only
    if not payload.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    if not payload.columns or not payload.rows:
        raise HTTPException(status_code=400, detail="Empty data")
    
    if len(payload.rows) > MAX_ROWS:
        raise HTTPException(status_code=400, detail=f"Max {MAX_ROWS:,} rows allowed")
    
    if len(payload.columns) > MAX_COLUMNS:
        raise HTTPException(status_code=400, detail=f"Max {MAX_COLUMNS} columns allowed")
    
    # Generate session
    session_id = str(uuid.uuid4())
    now = datetime.now()
    
    # Format file size
    size_bytes = payload.file_size
    if size_bytes < 1024:
        file_size_display = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        file_size_display = f"{size_bytes / 1024:.1f} KB"
    else:
        file_size_display = f"{size_bytes / (1024 * 1024):.1f} MB"
    
    # Store RAW data - no pandas!
    session_store[session_id] = {
        # Raw data (DataFrame created lazily in chat.py)
        "raw_columns": payload.columns,
        "raw_rows": payload.rows,
        "df": None,  # Created on first chat message
        
        # Metadata
        "file_name": payload.filename,
        "file_size": file_size_display,
        "upload_date": now.strftime("%Y-%m-%d"),
        "upload_time": now.strftime("%I:%M %p"),
        "columns": payload.columns,
        "num_rows": len(payload.rows),
        "num_columns": len(payload.columns),
        "preview_rows": payload.rows[:5],  # Just slice, no conversion
    }
    
    return JSONResponse({
        "success": True,
        "session_id": session_id,
        "rows": len(payload.rows),
        "columns": len(payload.columns),
    })

