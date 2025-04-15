import json
import numpy as np
import pandas as pd
from .file_path import data

def detect_outliers(df):
    """Detects outliers using IQR and Modified Z-score methods and returns outlier values."""
    outlier_summary = {}

    for col in df.select_dtypes(include=["number"]).columns:
        data = df[col].dropna()

        # IQR Method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        iqr_lower = Q1 - 1.5 * IQR
        iqr_upper = Q3 + 1.5 * IQR
        iqr_outliers = data[(data < iqr_lower) | (data > iqr_upper)]

        # Modified Z-score Method
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            z_outliers = pd.Series()  # Empty Series if no variability
        else:
            modified_z_scores = 0.6745 * (data - median) / mad
            z_outliers = data[np.abs(modified_z_scores) > 3.5]

        # Total Outlier Percentage
        total_outlier_percentage = round(((len(iqr_outliers) + len(z_outliers)) / len(data)) * 100, 2)

        outlier_summary[col] = {
            "iqr_outliers": iqr_outliers.tolist(),
            "modified_z_outliers": z_outliers.tolist(),
            "total_outlier_percentage": total_outlier_percentage,
        }

    return outlier_summary




def data_information(df):
    """Generates a structured dataset summary for AI processing."""

    num_rows, num_columns = df.shape
    outlier_summary = detect_outliers(df)

    # Duplicate rows
    duplicate_rows = df[df.duplicated(keep=False)].sort_values(by=list(df.columns))

    # Missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / num_rows * 100).round(3)

    column_details = {
        col: {
            "dtype": str(df[col].dtype),
            "unique_values": int(df[col].nunique()),
            "missing_count": int(missing_values[col]),
            "missing_percentage": float(missing_percentage[col]),
        }
        for col in df.columns
    }

    numerical_summary = df.describe().to_dict()

    context = {
        "dataset_size": {"rows": num_rows, "columns": num_columns},
        "column_details": column_details,
        "duplicate_rows": duplicate_rows.to_dict(orient='records'), #send the actual rows.
        "missing_values_summary": {col: {"missing_count": int(missing_values[col]), "missing_percentage": float(missing_percentage[col])} for col in df.columns},
        "numerical_summary": numerical_summary,
        "outlier_summary": outlier_summary,
    }

    return context



# Convert to JSON string for API request
json_dataset_context = json.dumps(data_information(data), indent=4)
json_dataset_context
