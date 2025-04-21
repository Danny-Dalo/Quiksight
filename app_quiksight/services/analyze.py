


from fastapi import APIRouter, UploadFile, File
import pandas as pd
import io, json
from ..services.data_overview import get_data_overview, _clean_value_for_json
from ..services.missing_data_analysis import analyze_missing_data
from api_training2.analyze_dataframe import analyze_larger_dataframe

router = APIRouter()

# @router.post("/analyze")
# async def analyze_data(file: UploadFile = File(...)):
#     # 1) We set default values for the objects that will be returned so that at every point they will always give at least a controlled error message
#     overview = {"error": "Overview did not run"}
#     missing_data = {"error": "Missing-data analysis did not run"}
#     gemini_result = {"error": "AI analysis did not run"}

#     # 2) dataframe is set to none by default and will be used later
#     df = None
#     encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']



#     # FILE READING SECTION
#     # ===================================================================================================
#     try:
#         # pauses the function and reads the file content as bytes into memory
#         raw = await file.read() 
        

#         # loops through available encodings and reads according to extension
#         for enc in encodings:
#             try:
#                 if file.filename.lower().endswith('.csv'):
#                     # The raw bytes are decoded into a string using the current (raw.decode)
#                     # io creates an in-memory bufferr(temp storage) that pandas can read from
#                     # StringIO (for CSV) stores a string to be read BytesIO (for excel) stores the excel to be read
#                     df = pd.read_csv(io.StringIO(raw.decode(enc)))
#                 else:  # .xls or .xlsx
#                     df = pd.read_excel(io.BytesIO(raw))
#                 print(f"[DEBUG] Loaded {file.filename} with encoding {enc}")
#                 break
#             except UnicodeDecodeError:
#                 continue
#         if df is None or df.empty:
#             return {"error": "Failed to read file or file is empty."}
#     except Exception as e:
#         return {"error": f"File read failed: {e}"}
#     # ======================================================================================

    
    

#     # AI ANALYSIS SECTION
#     # ==================================================================================================
#     try:
#         ai_analysis = analyze_larger_dataframe(df)

#         # Checks if the ai analysis is an instance of a string, it loads it as a JSON(that's how we want it) if it is
#         if isinstance(ai_analysis, str):        
#             ai_analysis = json.loads(ai_analysis)

#         #  If ai_analysis is neither a string nor a dictionary, it's considered to be in an invalid format.
#         elif not isinstance(ai_analysis, dict): 
#             ai_analysis = {"error": "Invalid AI result format"}

#         # After the checks and proper conversion if the ai analysis, it is passed to gemini_result variable
#         gemini_result = ai_analysis

    
#     except ZeroDivisionError: # This error is so subtle and caused a lot if issues
#         gemini_result = {"error": "AI analysis failed: division by zero"}
#     except Exception as e:
#         gemini_result = {"error": f"AI analysis failed: {e}"}
#         # ==================================================================================================



#     # DATA INFORMATION SECTION
#     # =======================================================================================================
#     # DATA OVERVIEW ERROR HANDLING
#     try:
#         overview = get_data_overview(df)
#     except Exception as e:
#         overview = {"error": f"Overview failed: {e}"}

#     # Missing‑data ERROR HANDLING
#     try:
#         missing_data = analyze_missing_data(df)
#     except Exception as e:
#         missing_data = {"error": f"Missing-data analysis failed: {e}"}

#     # =================================================================================================


#     # Return everything (now guaranteed to exist)
#     return {
#         "overview": overview,
#         "missing_data": missing_data,
#         "gemini_result": gemini_result
#     }























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



async def run_ai_analysis(df: pd.DataFrame):
    try:
        ai_result = analyze_larger_dataframe(df)
        if isinstance(ai_result, str):
            ai_result = json.loads(ai_result)
        elif not isinstance(ai_result, dict):
            return {"error": "Invalid AI result format"}
        
        return {"ai_result": ai_result}
    except Exception as e:
        return {"error": f"AI analysis failed: {e}"}






















# @router.post("/analyze")
# async def analyze_data(file: UploadFile = File(...)):
#     # 1) We set default values for the objects that will be returned so that at every point they will always give at least a controlled error message
#     overview = {"error": "Overview did not run"}
#     missing_data = {"error": "Missing-data analysis did not run"}

#     # 2) dataframe is set to none by default and will be used later
#     global df
#     df = None
#     encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']



#     # FILE READING SECTION
#     # ===================================================================================================
#     try:
#         # pauses the function and reads the file content as bytes into memory
#         raw = await file.read() 
        

#         # loops through available encodings and reads according to extension
#         for enc in encodings:
#             try:
#                 if file.filename.lower().endswith('.csv'):
#                     # The raw bytes are decoded into a string using the current (raw.decode)
#                     # io creates an in-memory bufferr(temp storage) that pandas can read from
#                     # StringIO (for CSV) stores a string to be read BytesIO (for excel) stores the excel to be read
#                     df = pd.read_csv(io.StringIO(raw.decode(enc)))
#                 else:  # .xls or .xlsx
#                     df = pd.read_excel(io.BytesIO(raw))
#                 print(f"[DEBUG] Loaded {file.filename} with encoding {enc}")
#                 break
#             except UnicodeDecodeError:
#                 continue
#         if df is None or df.empty:
#             return {"error": "Failed to read file or file is empty."}
#     except Exception as e:
#         return {"error": f"File read failed: {e}"}
#     # ======================================================================================



#     # DATA INFORMATION SECTION
#     # =======================================================================================================
#     # DATA OVERVIEW ERROR HANDLING
#     try:
#         overview = get_data_overview(df)
#     except Exception as e:
#         overview = {"error": f"Overview failed: {e}"}

#     # Missing‑data ERROR HANDLING
#     try:
#         missing_data = analyze_missing_data(df)
#     except Exception as e:
#         missing_data = {"error": f"Missing-data analysis failed: {e}"}

#     # =================================================================================================


#     # Return everything (now guaranteed to exist)
#     return {
#         "overview": overview,
#         "missing_data": missing_data
#     }





# async def run_ai_analysis(file: UploadFile = File(...)):
#     ai_result = {"error": "Couldn't get ai result"}
#     try:
#         ai_result = analyze_larger_dataframe(df)

#         if isinstance(ai_result, str):
#             ai_result = json.loads(ai_result)
#         elif not isinstance(ai_result, dict):
#             return {"error" : "Invalid AI result format"}
        
#         return {"ai_result" : ai_result}

#     except ZeroDivisionError:
#         return {"error": "AI analysis failed: division by zero"}
#     except Exception as e:
#         return {"error" : f"AI analysis failed: {e}"}
    


    