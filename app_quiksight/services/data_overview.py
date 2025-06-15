from typing import Dict, List, Any
import pandas as pd
import numpy as np





def _clean_value_for_json(value):
    """Recursively cleans values in nested structures for JSON compatibility."""
    if isinstance(value, dict):
        return {k: _clean_value_for_json(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_clean_value_for_json(item) for item in value]
    # Replace non-finite float values (NaN, inf, -inf) with None (JSON null)
    elif isinstance(value, float) and not np.isfinite(value):
        return None
    # Convert numpy int/float types to standard Python types
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
         # Check again for finite after potential numpy type conversion
        return float(value) if np.isfinite(value) else None
    # Add handling for other numpy types if necessary (e.g., np.bool_)
    elif isinstance(value, np.bool_):
        return bool(value)
    # Handle pandas Timestamp (convert to ISO format string)
    elif isinstance(value, pd.Timestamp):
        return value.isoformat()
    else:
        return value







def get_data_overview(df: pd.DataFrame) -> Dict[str, Any]:
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    

    descriptive_stats_raw = _get_descriptive_stats(df)
    table_head_raw = df.head(10).to_dict(orient="records")

    # Clean the data for JSON serialization BEFORE returning
    cleaned_stats = _clean_value_for_json(descriptive_stats_raw)
    cleaned_head = _clean_value_for_json(table_head_raw) # Cleans list of dicts


    return {
        "shape": {
            "columns": df.shape[1],
            "rows": df.shape[0]
        },
        "contains_missing": bool(df.isnull().values.any()), # Already safe
        "column_data_types": _get_column_data_types(df),    # Already safe
        "descriptive_stats": cleaned_stats,                 # Cleaned version
        "table_head": cleaned_head,                         # Cleaned version
        "unique_values": _get_unique_values(df),            # Already safe
        "duplicate_count": int(df.duplicated().sum())       # Already safe
    }








def _get_column_data_types(df: pd.DataFrame) -> List[Dict[str, str]]:
    
    return [
        {"column_name": col, "data_type": str(dtype)}
        for col, dtype in df.dtypes.items()
    ]




def _get_unique_values(df: pd.DataFrame) -> List[Dict[str, Any]]:
    
    return [
        {"column_name": col, "unique_values": int(df[col].nunique())}
        for col in df.columns
    ]




# def _get_descriptive_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    
#     return df.describe().to_dict()


def _get_descriptive_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculates descriptive statistics.
    Note: The result might contain non-JSON compliant floats (NaN, inf).
    Cleaning happens in the main get_data_overview function.
    """
    # Select only numerical columns for describe()
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return {} # Return empty dict if no numeric columns
    return numeric_df.describe().to_dict()
