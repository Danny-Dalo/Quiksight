import json
import numpy as np
import pandas as pd




def detect_outliers(df):
    """Detects outliers using IQR and Modified Z-score methods and returns outlier values."""
    outlier_summary = {}

    for col in df.select_dtypes(include=["number"]).columns:
        data = df[col].dropna()

        if len(data) == 0:
            outlier_summary[col] = {
                "iqr_outliers": [],
                "modified_z_outliers": [],
                "total_outlier_percentage": 0.0,
            }
            continue

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






def generate_summary_text(context):
    summary_lines = []

    # Dataset size
    rows = context['dataset_size']['rows']
    cols = context['dataset_size']['columns']
    summary_lines.append(f"The uploaded dataset contains **{rows} rows** and **{cols} columns**.")

    summary_lines.append("\n### Column Overview:")
    for col, details in context['column_details'].items():
        line = f"- `{col}`: type = {details['dtype']}, unique = {details['unique_values']}, "
        line += f"missing = {details['missing_percentage']}%"
        summary_lines.append(line)

    # Duplicates
    num_duplicates = len(context['duplicate_rows'])
    if num_duplicates > 0:
        summary_lines.append(f"\nThe dataset contains **{num_duplicates} duplicate rows**.")
    else:
        summary_lines.append("\nNo duplicate rows were found.")

    # Missing summary
    summary_lines.append("\n### Missing Values Summary:")
    for col, mv in context["missing_values_summary"].items():
        if mv['missing_count'] > 0:
            summary_lines.append(
                f"- `{col}`: {mv['missing_count']} missing values ({mv['missing_percentage']}%)"
            )

    # Numerical summary (describe stats)
    num_stats = context['numerical_summary']
    if num_stats:
        summary_lines.append("\n### Numerical Column Statistics:")
        for col, stats in num_stats.items():
            if all(key in stats for key in ["mean", "std", "min", "max"]):
                mean = stats["mean"]
                std = stats["std"]
                min_val = stats["min"]
                max_val = stats["max"]

                # Check for Timestamp
                if isinstance(mean, pd.Timestamp):
                    mean = mean.strftime("%Y-%m-%d")
                    std = "N/A"
                else:
                    mean = round(mean, 3)
                    std = round(std, 3)

                summary_lines.append(
                    f"- `{col}`: mean = {mean}, std = {std}, min = {min_val}, max = {max_val}"
                )

    # Outlier Summary
    outlier_info = context["outlier_summary"]
    summary_lines.append("\n### Outliers Detected:")
    for col, info in outlier_info.items():
        percent = info["total_outlier_percentage"]
        if percent > 0:
            summary_lines.append(f"- `{col}`: {percent}% of values flagged as outliers")
    
    # Optional: Health summary
    summary_lines.append("\n### Overall Data Quality Summary:")
    missing_cols = [col for col, mv in context["missing_values_summary"].items() if mv["missing_percentage"] > 0]
    has_outliers = any(v["total_outlier_percentage"] > 5 for v in outlier_info.values())
    # status = "✅ Clean" if not missing_cols and not has_outliers else "🟡 Moderate Issues"
    # summary_lines.append(f"**Status**: {status}")

    return "\n".join(summary_lines)

