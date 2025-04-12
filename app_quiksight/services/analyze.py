import os
import pandas as pd
from ..services.data_overview import get_data_overview
from ..services.missing_data_analysis import analyze_missing_data
from fastapi import APIRouter, UploadFile, File

router = APIRouter()









# FUNCTION TO ANALYZE UPLOADED FILES
@router.post("/analyze")
async def analyze_data(file: UploadFile = File(...)):
    try:
        df = None
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

        for encoding in encodings:
            try:
                if file.filename.endswith('.csv'):
                    content = await file.read()
                    from io import StringIO
                    df = pd.read_csv(StringIO(content.decode(encoding)))

                    break
                elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                    content = await file.read()
                    from io import BytesIO
                    df = pd.read_excel(BytesIO(content))

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
            "overview": get_data_overview(df),
            "missing_data": analyze_missing_data(df)
        }

    except Exception as e:
        return {"error": f"analyze_data error: {str(e)}"}


