

from fastapi import APIRouter, UploadFile
import pandas as pd
import io, json
from ..services.data_overview import get_data_overview, _clean_value_for_json
from ..services.missing_data_analysis import analyze_missing_data
from api_training2.analyze_dataframe import analyze_larger_dataframe

router = APIRouter()


MAX_ROWS_ALLOWED = 100000


# This function reads the uploaded file and returns a pandas dataframe
def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    raw = file.file.read()

    for enc in encodings:
        try:
            if file.filename.lower().endswith('.csv'):
                return pd.read_csv(io.StringIO(raw.decode(enc)))
            else:
                return pd.read_excel(io.BytesIO(raw))
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode file")






async def analyze_data(file: UploadFile):
    try:
        df = read_uploaded_file(file)
    except Exception as e:
        return {"error": f"File read failed: {e}"}
    
    if len(df) > MAX_ROWS_ALLOWED:
        return {"error": f"File exceeds the {MAX_ROWS_ALLOWED} row limit. Current rows: {len(df)}"}

    try:
        overview = get_data_overview(df)
    except Exception as e:
        overview = {"error": f"Overview failed: {e}"}

    try:
        missing_data = analyze_missing_data(df)
    except Exception as e:
        missing_data = {"error": f"Missing-data analysis failed: {e}"}

    return {
        "overview": overview,
        "missing_data": missing_data,
        "df": df  # add this
    }







# ================================================================

async def run_ai_analysis(df: pd.DataFrame):
    try:
        cleaned_json_df = df.map(lambda x: _clean_value_for_json(x))

        ai_result = analyze_larger_dataframe(cleaned_json_df)

        if isinstance(ai_result, str):
            ai_result = ai_result.strip()
            if not ai_result:
                return {"error": "AI returned an empty response"}
            try:
                ai_result = json.loads(ai_result)
            except Exception as e:
                return {"error": f"AI response is not valid JSON: {ai_result[:200]}... Error: {e}"}

        elif not isinstance(ai_result, dict):
            return {"error": f"Invalid AI result format: {type(ai_result)}"}

        return {"ai_result": ai_result}
    except Exception as e:
        return {"error": f"AI analysis failed: {e}"}








