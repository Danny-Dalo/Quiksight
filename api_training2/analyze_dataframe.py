"""
AI-Powered Data Quality Analysis Module

This module provides AI-driven data quality analysis using Google's Gemini API.
It includes intelligent data sampling, context generation, and robust response parsing.
"""

import json
import time
import pandas as pd
import numpy as np

from .data_context import data_information
# from .data_dictionary import generate_data_dictionary
from .call_gemini import call_gemini_api, GEMINI_API_KEY

import datetime
def json_serializer(obj):
    """
    Custom JSON serializer for pandas and numpy objects.
    
    Handles Timestamp, NaT, numpy integers, floats, and arrays that aren't
    serializable by default JSON encoder (json.dumps()).
    
    Args:
        obj: Object to serialize
        
    Returns:
        Serialized object suitable for JSON encoding
        
    Raises:
        TypeError: If object type is not supported
    """
    if isinstance(obj, (pd.Timestamp, pd.NaT.__class__)):
        return obj.isoformat() if pd.notna(obj) else None
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


# DATA SAMPLING FUNCTION
def sample_dataframe_for_ai(df: pd.DataFrame, max_records: int = 100) -> list:
    """
    Intelligently sample a DataFrame for AI analysis.
    
    Creates a representative sample that includes:
    - Rows with missing values (to show data quality issues)
    - Most frequent and rarest values from each column
    - Random samples for variety
    
    Args:
        df (pd.DataFrame): DataFrame to sample
        max_records (int): Maximum number of records to include
        
    Returns:
        list: List of dictionaries representing sampled data
    """
    sampled_rows = set()
    
    # Step 1: Include rows with missing values (important for data quality analysis)
    nan_rows = df[df.isna().any(axis=1)]
    sampled_rows.update(nan_rows.index[:5])  # Add up to 5 rows with NaNs

    # Step 2: Sample representative values from each column
    for col in df.columns:
        # Include most frequent value
        top_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
        if top_val is not None:
            top_row_idx = df[df[col] == top_val].index[0]
            sampled_rows.add(top_row_idx)

        # Include rarest value (if different from most frequent)
        value_counts = df[col].value_counts()
        if len(value_counts) > 1:  # Only if there are multiple unique values
            rare_val = value_counts.index[-1]
            if rare_val != top_val:  # Avoid duplicates
                rare_row_idx = df[df[col] == rare_val].index[0]
                sampled_rows.add(rare_row_idx)

        # Add random samples for variety
        random_rows = df.sample(min(2, len(df)), random_state=42).index
        sampled_rows.update(random_rows)

    # Step 3: Fill remaining slots with random rows if needed
    if len(sampled_rows) < max_records:
        remaining = max_records - len(sampled_rows)
        available_rows = df.drop(index=list(sampled_rows))
        if len(available_rows) > 0:
            random_extra = available_rows.sample(
                min(remaining, len(available_rows)), random_state=42
            ).index
            sampled_rows.update(random_extra)

    # Convert to list of dictionaries for JSON serialization
    final_df = df.loc[list(sampled_rows)].copy()
    return final_df.to_dict(orient="records")









def analyze_larger_dataframe(df: pd.DataFrame, API_KEY: str = GEMINI_API_KEY, delay: int = 2) -> dict:
    """
    Perform comprehensive AI-powered data quality analysis.
    
    This function orchestrates the entire analysis process:
    1. Samples the data intelligently
    2. Generates dataset context
    3. Creates analysis prompt
    4. Calls AI API
    5. Parses and validates response
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        API_KEY (str): Gemini API key
        delay (int): Delay before API call (rate limiting)
        
    Returns:
        dict: Analysis results or error information
    """
    # Step 1: Create intelligent sample for AI analysis
    dataset_sample = sample_dataframe_for_ai(df, max_records=500)

    # Step 2: Generate dataset context and metadata
    json_dataset_context = data_information(df)
    if not json_dataset_context:
        print("Error: Failed to generate dataset context.")
        return None

    # Step 3: Create comprehensive analysis prompt
    prompt = _create_analysis_prompt(dataset_sample, json_dataset_context)

    # Step 4: Call AI API with error handling
    try:
        time.sleep(delay)  # Rate limiting
        response = call_gemini_api(prompt, api_key=API_KEY)
        
        if not response:
            print("Empty API response")
            return None
        
        response = response.strip()
        
        # Step 5: Parse and validate AI response
        return _parse_ai_response(response)
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return None





















def _create_analysis_prompt(dataset_sample: list, json_dataset_context: dict) -> str:
    """
    Create a comprehensive prompt for AI data quality analysis.
    
    Args:
        dataset_sample (list): Sampled data for analysis
        json_dataset_context (dict): Dataset metadata and context
        
    Returns:
        str: Formatted prompt for AI analysis
    """
    return f"""
You are a Data Quality Specialist with expertise in **business intelligence, data cleaning, and governance**.  
You will analyze the dataset below and report findings as if assisting a **professional data scientist**.  

Your job is to identify **real, context-aware data quality issues** — not just syntactic issues, but **semantic, factual, and structural ones**.  
Examples include: misspellings, invalid mappings (e.g., wrong continent for a country), unrealistic values, unexpected/improper formats, inconsistencies and other anomalies that may be hard to spot.

---
### 🔍 Dataset Inputs:
- **Sample Data**: {json.dumps(dataset_sample, indent=4, default=json_serializer)}

- **Metadata/Context**: {json.dumps(json_dataset_context, default=json_serializer)}

---
### Analysis Objectives:

1. **High-Level Summary**
   - Summarize what the dataset is about in 2-3 clear sentences.
   - Highlight important columns and structural characteristics (e.g., ID columns, date ranges, groupings).

2. **Identify & Categorize Issues**
   Review all columns in context and list issues under:
   - **Critical Issues**: Severe problems (e.g., factual inaccuracies, broken keys, nulls in key columns).
   - **Moderate Issues**: Likely to affect analysis (e.g., inconsistent formats, suspicious outliers).
   - **Minor Issues**: Mild inconsistencies (e.g., minor spelling errors, naming variations).

   ✔ Think critically. Only report issues that **matter in real-world usage**.  
    Do **not** make up issues just to fill space. If data is clean, state it confidently.

   - **Date-Time Quality Check**
  - Review all date-like fields for:
    - Inconsistent or invalid formats (e.g., mixed `YYYY-MM-DD`, `DD/MM/YYYY`)
    - Wrong data types (e.g., string instead of datetime)
    - Unrealistic values (e.g., 1800s, 2100s, etc.)
    - Misaligned date logic (e.g., future birth dates, negative durations)

3. **Business Impact & Fix Suggestions**
   - For each issue, include:
     - **Impact**: What effect could this have on business analysis or decision-making?
     - **Recommendation**: What is the best fix for this specific issue? (Be clear and actionable)

---
### Response Format Requirements:

- Return only **clean JSON**. Do NOT include markdown formatting like `**bold**`, `*italic*`, or code fences (no ```json or ```).
- For all `"column"` values, return them as a **comma-separated string** if multiple columns are affected — e.g., `"column": "reporter name, partner name_adj"` instead of a list like `["..."]`.
- Keep all values as simple strings — no lists, markdown, or extra syntax.
- Each recommendation should be a simple string, no bullet points or asterisks.
- Avoid any unnecessary formatting or noise — return raw, readable, clean JSON only.

### Expected JSON Structure:
{{
    "summary": "Short, clear overview of the dataset.",
    "issues": {{
        "critical": [
            {{
                "column": "ColumnName",
                "issue": "What is wrong and why it's a problem.",
                "impact": "What this could affect in real-world use.",
                "recommendation": "How to fix this issue."
            }}
        ],
        "moderate": [{{
                "column": "ColumnName",
                "issue": "What is wrong and why it's a problem.",
                "impact": "What this could affect in real-world use.",
                "recommendation": "How to fix this issue."
        }}],
        "minor": [{{
                "column": "ColumnName",
                "issue": "What is wrong and why it's a problem.",
                "impact": "What this could affect in real-world use.",
                "recommendation": "How to fix this issue."
        }}]
    }}
}}
"""














def _parse_ai_response(response: str) -> dict:
    """
    Parse and validate AI response with robust error handling.
    
    Handles various response formats including:
    - Clean JSON
    - JSON wrapped in markdown code blocks
    - JSON with extra text before/after
    - Malformed JSON that needs extraction
    
    Args:
        response (str): Raw response from AI API
        
    Returns:
        dict: Parsed response or structured error response
    """
    try:
        # Step 1: Remove markdown code blocks if present
        response = _remove_markdown_blocks(response)
        
        # Step 2: Try direct JSON parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Step 3: If direct parsing fails, try to extract JSON object
            return _extract_json_object(response)
            
    except json.JSONDecodeError as e:
        # Step 4: If all parsing attempts fail, return structured error
        return _create_parse_error_response(response, str(e))


def _remove_markdown_blocks(response: str) -> str:
    """
    Remove markdown code block formatting from response.
    
    Args:
        response (str): Response that may contain markdown formatting
        
    Returns:
        str: Response with markdown blocks removed
    """
    if response.startswith('```'):
        lines = response.split('\n')
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        response = '\n'.join(lines)
    return response


def _extract_json_object(response: str) -> dict:
    """
    Extract JSON object from response that may contain extra text.
    
    Args:
        response (str): Response that may contain JSON object
        
    Returns:
        dict: Extracted JSON or structured error response
    """
    start = response.find('{')
    end = response.rfind('}') + 1
    
    if start != -1 and end > start:
        json_str = response[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # JSON extraction failed
            return _create_parse_error_response(response, "Failed to parse extracted JSON")
    else:
        # No JSON object found
        return _create_parse_error_response(response, "No JSON object found in response")


def _create_parse_error_response(response: str, error_msg: str) -> dict:
    """
    Create a structured error response when JSON parsing fails.
    
    Args:
        response (str): Raw AI response
        error_msg (str): Description of parsing error
        
    Returns:
        dict: Structured error response
    """
    return {
        "summary": "AI analysis completed but response could not be parsed as JSON",
        "issues": {
            "critical": [],
            "moderate": [],
            "minor": []
        },
        "recommendations": ["Please review the raw AI response for manual analysis"],
        "raw_response": response,
        # "raw_response": response[:500] + "..." if len(response) > 500 else response,
        "parse_error": error_msg
    }

