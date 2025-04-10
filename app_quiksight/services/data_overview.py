import pandas as pd

def get_data_overview(df: pd.DataFrame) -> dict:
    """Returns comprehensive overview of the DataFrame."""
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return {
        "shape": {
            "columns": df.shape[1],
            "rows": df.shape[0]
        },
        "contains_missing": "Yes" if df.isnull().sum().any() else "No",
        "column_data_types": _generate_data_types_table(df),
        "descriptive_stats": df.describe().to_html(classes='table table-bordered'),
        "table_head": df.head(10).to_html(classes='table table-striped'),
        "unique_values": _generate_unique_values_table(df),
        "duplicate_count": df.duplicated().sum()
    }

def _generate_data_types_table(df: pd.DataFrame) -> str:
    return pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str)
    }).to_html(index=False, classes='table table-bordered')

def _generate_unique_values_table(df: pd.DataFrame) -> str:
    return pd.DataFrame({
        "Column Name": df.columns,
        "Unique Values": df.nunique()
    }).to_html(index=False, classes='table table-bordered')