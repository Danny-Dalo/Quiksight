import pandas as pd
from typing import Dict, Any

# Dict: Indicates that the function will return a dictionary.
# Any: Indicates that the values within the dictionary can be of any type.
def analyze_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
    # return a dictionary where keys are strings and values can be of any type.

    total = len(df)
    # guard against zero‑row DataFrame

    # return a dictionary with key as "missing data" and values as an empty list, ensuring something is still returned
    if total == 0:
        return {"missing_data": []}

    # =================================================================================
    missing_columns_data = []

    for col in df.columns:
        missing = int(df[col].isnull().sum())

        if missing > 0:  # Only process columns with missing data
            # safe division
            pct = round((missing / total) * 100, 2)
            missing_columns_data.append({
                "column": col,
                "missing_count": missing,
                "missing_percentage": pct
            })
            

    if not missing_columns_data:
        return {"missing_data" : "No missing data found"}
    return {"missing_data": missing_columns_data}

        
        

