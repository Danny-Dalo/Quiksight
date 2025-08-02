# """
# Data Analysis Service Module

# This module handles file uploads, data processing, and AI-powered data quality analysis.
# It provides functions to read various file formats, clean data, and generate insights.
# """

from fastapi import APIRouter, UploadFile
import pandas as pd
import io
import csv
import numpy as np
import json
from ..services.data_overview import get_data_overview, _clean_value_for_json
from ..services.missing_data_analysis import analyze_missing_data
from api_training2.data_context import generate_summary_text, data_information

router = APIRouter()

# # Configuration
MAX_ROWS_ALLOWED = 100000
SUPPORTED_ENCODINGS = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']


def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    """
    Read and parse uploaded files into pandas DataFrames.
    
    Supports CSV and Excel files with multiple encoding and engine fallbacks.
    
    Args:
        file (UploadFile): The uploaded file object
        
    Returns:
        pd.DataFrame: Parsed data as a pandas DataFrame
        
    Raises:
        ValueError: If file cannot be read or parsed
    """
    # Load file content into memory as bytes
    raw = file.file.read()

    try:
        if file.filename.lower().endswith('.csv'):
            return _read_csv_file(raw)
        else:
            return _read_excel_file(raw)
    except Exception as e:
        raise ValueError(f"File reading failed: {str(e)}")


def _read_csv_file(raw: bytes) -> pd.DataFrame:
    """
    Read CSV file with multiple encoding attempts.
    
    Args:
        raw (bytes): Raw file content
        
    Returns:
        pd.DataFrame: Parsed CSV data
        
    Raises:
        ValueError: If no encoding works
    """
    for encoding in SUPPORTED_ENCODINGS:
        try:
            return pd.read_csv(
                io.StringIO(raw.decode(encoding)),
                engine="python",  # <-- Switch from pyarrow to python
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
                skip_blank_lines=True,
            )
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode CSV file with any supported encoding")


def _read_excel_file(raw: bytes) -> pd.DataFrame:
    """
    Read Excel file with multiple engine fallbacks.
    
    Args:
        raw (bytes): Raw file content
        
    Returns:
        pd.DataFrame: Parsed Excel data
        
    Raises:
        ValueError: If no engine can read the file
    """
    engines_to_try = ['openpyxl', 'xlrd', None]  # None = pandas default
    
    for engine in engines_to_try:
        try:
            if engine:
                return pd.read_excel(io.BytesIO(raw), engine=engine)
            else:
                return pd.read_excel(io.BytesIO(raw))
        except Exception as e:
            last_error = e
            continue
    
    raise ValueError(f"Could not read Excel file with any engine. Last error: {last_error}")


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the DataFrame for analysis.
    
    Args:
        df (pd.DataFrame): Raw DataFrame to clean
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
        
    Raises:
        ValueError: If DataFrame becomes empty after cleaning
    """
    # Remove completely empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Validate that we still have data
    if df.empty:
        raise ValueError("File contains no valid data after cleaning")
    
    # Ensure column names are strings and clean them
    df.columns = df.columns.astype(str)
    
    # Replace problematic characters in column names with underscores
    df.columns = df.columns.str.replace('[^\w\s-]', '_', regex=True)
    
    return df




from datetime import datetime, date

def clean_value_for_json(value):
    """Clean a value to make it JSON serializable"""
    if pd.isna(value) or value is None:
        return None
    elif isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    elif isinstance(value, (datetime, date)):
        return value.isoformat()
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    elif isinstance(value, str):
        return value
    else:
        return str(value)

def truncate_sample_row_values(rows: list, max_len: int = 100) -> list:
    """Truncate and clean sample row values for JSON serialization"""
    truncated_rows = []
    for row in rows:
        truncated_row = {}
        for key, value in row.items():
            # First clean the value for JSON serialization
            cleaned_value = clean_value_for_json(value)
            
            # Then truncate if it's a string
            if isinstance(cleaned_value, str) and len(cleaned_value) > max_len:
                truncated_row[key] = cleaned_value[:max_len] + "..."
            else:
                truncated_row[key] = cleaned_value
        truncated_rows.append(truncated_row)
    return truncated_rows



async def analyze_data(file: UploadFile) -> dict:
    """
    Main function to analyze uploaded data files.
    
    Performs file reading, data cleaning, overview generation, and missing data analysis.
    
    Args:
        file (UploadFile): The uploaded file to analyze
        
    Returns:
        dict: Analysis results containing overview, missing data info, and cleaned DataFrame
    """
    # Step 1: Read the uploaded file
    try:
        df = read_uploaded_file(file)
    except Exception as e:
        return {"error": f"File read failed: {e}"}
    
    # Check file size
    if len(df) > MAX_ROWS_ALLOWED:
        return {
            "error": f"File exceeds the {MAX_ROWS_ALLOWED:,} row limit. Current rows: {len(df):,}"
        }

    # Clean and validate the data
    try:
        df = _clean_dataframe(df)
    except Exception as e:
        return {"error": f"Data cleaning failed: {e}"}

    # Generate data overview
    try:
        overview = get_data_overview(df)
    except Exception as e:
        overview = {"error": f"Overview generation failed: {e}"}

    # Analyze missing data
    try:
        missing_data = analyze_missing_data(df)
    except Exception as e:
        missing_data = {"error": f"Missing data analysis failed: {e}"}


        # Context to give the model
    context = data_information(df)
    
    # ==================================================================================
    sample_rows = df.sample(min(len(df), 5)).to_dict(orient="records")
    sample_rows = truncate_sample_row_values(sample_rows)
    # ==================================================================================

    return {
        "overview": overview,
        "missing_data": missing_data,
        "data_summary" : generate_summary_text(context),
        # ==================================================================================
        "sample_rows" : sample_rows,
        # ==================================================================================
        "df": df
    }
    