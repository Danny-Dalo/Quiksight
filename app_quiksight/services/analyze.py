

from fastapi import APIRouter, UploadFile
import pandas as pd
import io, json
from ..services.data_overview import get_data_overview, _clean_value_for_json
from ..services.missing_data_analysis import analyze_missing_data
from api_training2.analyze_dataframe import analyze_larger_dataframe

router = APIRouter()


MAX_ROWS_ALLOWED = 100000


# This function reads the uploaded file and returns it as a pandas dataframe
def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    #loads content into RAM as bytes
    raw = file.file.read()

    for enc in encodings:
        try:
            if file.filename.lower().endswith('.csv'):
                # Wraps it in a StringIO object so pd.read_csv can read it like a file.
                return pd.read_csv(io.StringIO(raw.decode(enc)))
            else:
                return pd.read_excel(io.BytesIO(raw))
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode file")





# This function reads uploaded file into a dataframe(via the above function), it returns the dataframe, the data overview, and missing value information of the data
async def analyze_data(file: UploadFile):

    # calls function to read uploaded file, if for any reason the file read failed it catches the error
    try:
        df = read_uploaded_file(file)
    except Exception as e:
        return {"error": f"File read failed: {e}"}
    
    # Validates for max row limit
    if len(df) > MAX_ROWS_ALLOWED:
        return {"error": f"File exceeds the {MAX_ROWS_ALLOWED} row limit. Current rows: {len(df)}"}

    # calls function to get overview of data, it fails if there is an error with the overview function
    try:
        overview = get_data_overview(df)
    except Exception as e:
        overview = {"error": f"Overview failed: {e}"}

    # calls function to get missing info from data, it fails if there is an error with the function
    try:
        missing_data = analyze_missing_data(df)
    except Exception as e:
        missing_data = {"error": f"Missing-data analysis failed: {e}"}

    return {
        "overview": overview,
        "missing_data": missing_data,
        "df": df  
    }







# ================================================================

# Takes in dataframe as input, no longer file, it performs a series of formatting to make sure AI-generated output is cleaned and in a good format. it returns the AI output as a dictionary
async def run_ai_analysis(df: pd.DataFrame):

    try:
        # applies jdon cleaning to every value in dataframe
        cleaned_json_df = df.map(lambda x: _clean_value_for_json(x))

        # sends the cleaned json dataframe to the function for AI to do it's thing
        ai_result = analyze_larger_dataframe(cleaned_json_df)

        # if the AI returns a string(the common response type), it removes whitespaces from it
        if isinstance(ai_result, str):
            ai_result = ai_result.strip()
            # shows an error if AI result is empty
            if not ai_result:
                return {"error": "AI returned an empty response"}
            
            # converts the AI output into a python dictionary, this is because we told our model to generate output formatted as a dictionary because the AI's output its still just a string that just LOOKS like a dictionary
            try:
                ai_result = json.loads(ai_result)
            except Exception as e:
                return {"error": f"AI response is not valid JSON: {ai_result[:200]}... Error: {e}"} # shows just 200 characters of the error

        
        elif not isinstance(ai_result, dict):
            return {"error": f"Invalid AI result format: {type(ai_result)}"}

        return {"ai_result": ai_result}
    
    except Exception as e:
        return {"error": f"AI analysis failed: {e}"}








