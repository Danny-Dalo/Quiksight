"""
Data Analysis Service Module

This module handles file uploads, data processing, and AI-powered data quality analysis.
It provides functions to read various file formats, clean data, and generate insights.
"""

from fastapi import APIRouter, UploadFile
import pandas as pd
import io
import json
from ..services.data_overview import get_data_overview, _clean_value_for_json
from ..services.missing_data_analysis import analyze_missing_data
from api_training2.analyze_dataframe import analyze_larger_dataframe

router = APIRouter()

# Configuration
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
            return pd.read_csv(io.StringIO(raw.decode(encoding)))
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
    
    # Step 2: Validate file size
    if len(df) > MAX_ROWS_ALLOWED:
        return {
            "error": f"File exceeds the {MAX_ROWS_ALLOWED:,} row limit. Current rows: {len(df):,}"
        }

    # Step 3: Clean and validate the data
    try:
        df = _clean_dataframe(df)
    except Exception as e:
        return {"error": f"Data cleaning failed: {e}"}

    # Step 4: Generate data overview
    try:
        overview = get_data_overview(df)
    except Exception as e:
        overview = {"error": f"Overview generation failed: {e}"}

    # Step 5: Analyze missing data
    try:
        missing_data = analyze_missing_data(df)
    except Exception as e:
        missing_data = {"error": f"Missing data analysis failed: {e}"}

    return {
        "overview": overview,
        "missing_data": missing_data,
        "df": df
    }


async def run_ai_analysis(df: pd.DataFrame) -> dict:
    """
    Run AI-powered data quality analysis on the DataFrame.
    
    This function prepares the data for AI analysis and handles various response formats
    from the AI model, ensuring robust error handling.
    
    Args:
        df (pd.DataFrame): Clean DataFrame to analyze
        
    Returns:
        dict: AI analysis results or error information
    """
    try:
        # Step 1: Clean data for JSON serialization
        cleaned_df = df.map(lambda x: _clean_value_for_json(x))
        
        # Step 2: Send to AI for analysis
        ai_result = analyze_larger_dataframe(cleaned_df)
        
        # Step 3: Process AI response
        return _process_ai_response(ai_result)
        
    except Exception as e:
        return {"error": f"AI analysis failed: {e}"}


def _process_ai_response(ai_result) -> dict:
    """
    Process and validate the AI response, handling various formats.
    
    Args:
        ai_result: Raw response from AI analysis
        
    Returns:
        dict: Processed AI result or structured error response
    """
    # Handle string responses (most common)
    if isinstance(ai_result, str):
        return _process_string_response(ai_result)
    
    # Handle dictionary responses (already processed)
    elif isinstance(ai_result, dict):
        return {"ai_result": ai_result}
    
    # Handle unexpected response types
    else:
        return _create_error_response(
            f"AI analysis completed but returned unexpected format: {type(ai_result)}",
            str(ai_result)
        )


def _process_string_response(ai_result: str) -> dict:
    """
    Process AI response that comes as a string.
    
    Args:
        ai_result (str): Raw string response from AI
        
    Returns:
        dict: Processed result or error response
    """
    # Clean the response
    ai_result = ai_result.strip()
    
    # Check for empty response
    if not ai_result:
        return {"error": "AI returned an empty response"}
    
    # Try to parse as JSON
    try:
        parsed_result = json.loads(ai_result)
        return {"ai_result": parsed_result}
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON
        return _extract_json_from_response(ai_result)


def _extract_json_from_response(response: str) -> dict:
    """
    Extract JSON object from a response that may contain extra text.
    
    Args:
        response (str): Response that may contain JSON
        
    Returns:
        dict: Extracted JSON or error response
    """
    try:
        # Find JSON object boundaries
        start = response.find('{')
        end = response.rfind('}') + 1
        
        if start != -1 and end > start:
            json_str = response[start:end]
            parsed_result = json.loads(json_str)
            return {"ai_result": parsed_result}
        else:
            # No JSON object found
            return _create_error_response(
                "AI analysis completed but response format was unexpected",
                response
            )
    except json.JSONDecodeError as e:
        # JSON extraction failed
        return _create_error_response(
            "AI analysis completed but response could not be parsed as JSON",
            response,
            parse_error=str(e)
        )


def _create_error_response(summary: str, raw_response: str, parse_error: str = None) -> dict:
    """
    Create a structured error response when AI analysis fails.
    
    Args:
        summary (str): Summary of what went wrong
        raw_response (str): Raw AI response for debugging
        parse_error (str, optional): Specific parsing error details
        
    Returns:
        dict: Structured error response
    """
    error_response = {
        "summary": summary,
        "issues": {
            "critical": [],
            "moderate": [],
            "minor": []
        },
        "recommendations": ["Please review the raw AI response for manual analysis"],
        "raw_response": raw_response[:500] + "..." if len(raw_response) > 500 else raw_response
    }
    
    if parse_error:
        error_response["parse_error"] = parse_error
    
    return {"ai_result": error_response}








