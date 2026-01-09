"""
Upload route - Receives PRE-PARSED data from client-side JavaScript.
The client parses CSV/Excel files using SheetJS and sends JSON data.
This reduces CPU load on the server (Render free tier).
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='\n%(asctime)s | %(levelname)-8s | %(name)-10s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("UPLOAD")

router = APIRouter()

# In-memory session storage
session_store: Dict[str, Dict[str, Any]] = {}

# Constants
MAX_ROWS = 100_000  # Max rows allowed
MAX_COLUMNS = 500   # Max columns allowed


class UploadPayload(BaseModel):
    """Schema for client-side parsed data."""
    filename: str
    file_size: int  # in bytes
    columns: List[str]
    rows: List[Dict[str, Any]]  # Array of row objects


@router.post("/upload")
async def upload_parsed_data(payload: UploadPayload):
    """
    Receive pre-parsed data from client-side JavaScript.
    Client uses SheetJS to parse CSV/Excel files before sending.
    """
    logger.info(f"📤 Received parsed data: {payload.filename}")
    logger.info(f"   Rows: {len(payload.rows)}, Columns: {len(payload.columns)}")
    
    # Validate payload
    if not payload.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    if not payload.columns:
        raise HTTPException(status_code=400, detail="No columns found in data")
    
    if not payload.rows:
        raise HTTPException(status_code=400, detail="No data rows found")
    
    if len(payload.rows) > MAX_ROWS:
        raise HTTPException(
            status_code=400, 
            detail=f"Too many rows. Maximum allowed: {MAX_ROWS:,}"
        )
    
    if len(payload.columns) > MAX_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many columns. Maximum allowed: {MAX_COLUMNS}"
        )
    
    try:
        # Convert to DataFrame (lightweight since data is already parsed)
        df = pd.DataFrame(payload.rows, columns=payload.columns)
        logger.info(f"   ✓ DataFrame created: {df.shape}")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Format file size for display
        size_bytes = payload.file_size
        if size_bytes < 1024:
            file_size_display = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            file_size_display = f"{size_bytes / 1024:.2f} KB"
        else:
            file_size_display = f"{size_bytes / (1024 * 1024):.2f} MB"
        
        # Get current timestamp
        now = datetime.now()
        
        # Store session data
        session_store[session_id] = {
            "df": df,
            "file_name": payload.filename,
            "file_size": file_size_display,
            "upload_date": now.strftime("%Y-%m-%d"),
            "upload_time": now.strftime("%I:%M %p"),
            "columns": payload.columns,
            "preview_rows": df.head(5).to_dict(orient="records"),
            "num_rows": len(df),
            "num_columns": len(payload.columns),
        }
        
        logger.info(f"   ✓ Session created: {session_id[:8]}...")
        logger.info(f"   Active sessions: {len(session_store)}")
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "rows": len(df),
            "columns": len(payload.columns),
            "preview": df.head(5).to_dict(orient="records"),
        })
        
    except Exception as e:
        logger.error(f"❌ Error processing data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process data: {str(e)}")
