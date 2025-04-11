import os
import pandas as pd
from . import data_overview
from . import missing_data_analysis



def analyze_data(file_path):
    try:
        df = None
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

        for encoding in encodings:
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                    df = pd.read_excel(file_path)
                    break
                else:
                    return {"error": "Unsupported file format"}
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return {"error": str(e)}

        if df is None:
            return {"error": "Failed to read file. Try another encoding or format."}

        return {
            "overview": data_overview.get_data_overview(df),
            "missing_data": missing_data_analysis.analyze_missing_data(df) if hasattr(missing_data_analysis, 'analyze_missing_data') else {}
        }

    except Exception as e:
        return {"error": f"analyze_data error: {str(e)}"}
