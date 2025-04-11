from typing import Dict, List, Any
import pandas as pd

def get_data_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive overview of the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to analyze
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - shape: Dict with number of rows and columns
            - contains_missing: bool indicating presence of missing values
            - column_data_types: List of column names and their types
            - descriptive_stats: Statistical summary of numerical columns
            - table_head: First 10 rows of data
            - unique_values: Count of unique values per column
            - duplicate_count: Number of duplicate rows
            
    Raises:
        ValueError: If DataFrame is empty
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    return {
        "shape": {
            "columns": df.shape[1],
            "rows": df.shape[0]
        },
        "contains_missing": df.isnull().values.any(),
        "column_data_types": _get_column_data_types(df),
        "descriptive_stats": _get_descriptive_stats(df),
        "table_head": df.head(10).to_dict(orient="records"),
        "unique_values": _get_unique_values(df),
        "duplicate_count": int(df.duplicated().sum())
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




def _get_descriptive_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    
    return df.describe().to_dict()

