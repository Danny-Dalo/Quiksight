# import os
# import pandas as pd
# from ..services.data_overview import get_data_overview
# from ..services.missing_data_analysis import analyze_missing_data
# from fastapi import APIRouter, UploadFile, File
# from io import StringIO, BytesIO
# from api_training2.analyze_dataframe import analyze_larger_dataframe
# import json


# router = APIRouter()









# FUNCTION TO ANALYZE UPLOADED FILES
# @router.post("/analyze")
# async def analyze_data(file: UploadFile = File(...)):

#     try:
#         df = None
#         encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

#         for encoding in encodings:
#             try:

#                 if file.filename.endswith('.csv'):
#                     content = await file.read()
#                     from io import StringIO
#                     df = pd.read_csv(StringIO(content.decode(encoding)))

#                     break
#                 elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
#                     content = await file.read()
#                     from io import BytesIO
#                     df = pd.read_excel(BytesIO(content))

#                     break
#                 else:
#                     return {"error": "Unsupported file format"}
#             except UnicodeDecodeError:
#                 continue
#             except Exception as e:
#                 return {"error": str(e)}
            
# # ==============================================================================
#         if df is None:
#             return {"error": "Failed to read file. Try another encoding or format."}
#         # ==============================================================================

        
#         # ==============================================================================

#         gemini_result = analyze_larger_dataframe(df)
#         if gemini_result:
#             try:
#                 # If it's already a dict, use it directly
#                 if isinstance(gemini_result, dict):
#                     pass
#                 # If it's a string, try to parse it as JSON
#                 elif isinstance(gemini_result, str):
#                     gemini_result = json.loads(gemini_result)
#                 else:
#                     gemini_result = {"error": "Invalid result format"}
#             except json.JSONDecodeError as e:
#                 print(f"Error parsing Gemini result: {str(e)}")
#                 gemini_result = {"error": "Error processing AI analysis results"}
#             except Exception as e:
#                 print(f"Error processing Gemini result: {str(e)}")
#                 gemini_result = {"error": "Error processing AI analysis results"}


#         return {
#             "overview": get_data_overview(df),
#             "missing_data": analyze_missing_data(df),
#             "gemini_result": gemini_result
#         }

#     except Exception as e:
#         return {"error": f"analyze_data error: {str(e)}"}









from fastapi import APIRouter, UploadFile, File
import pandas as pd
import io, json
from ..services.data_overview import get_data_overview
from ..services.missing_data_analysis import analyze_missing_data
from api_training2.analyze_dataframe import analyze_larger_dataframe

router = APIRouter()

@router.post("/analyze")
async def analyze_data(file: UploadFile = File(...)):
    # 1) We set default values for the objects that will be returned so that at every point they will always give at least a controlled error message
    overview = {"error": "Overview did not run"}
    missing_data = {"error": "Missing-data analysis did not run"}
    gemini_result = {"error": "AI analysis did not run"}

    # 2) dataframe is set to none by default and will be used later
    df = None
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']



    # FILE READING SECTION
    # ===================================================================================================
    try:
        # pauses the function and reads the file content as bytes into memory
        raw = await file.read() 
        

        # loops through available encodings and reads according to extension
        for enc in encodings:
            try:
                if file.filename.lower().endswith('.csv'):
                    # The raw bytes are decoded into a string using the current (raw.decode)
                    # io creates an in-memory bufferr(temp storage) that pandas can read from
                    # StringIO (for CSV) stores a string to be read BytesIO (for excel) stores the excel to be read
                    df = pd.read_csv(io.StringIO(raw.decode(enc)))
                else:  # .xls or .xlsx
                    df = pd.read_excel(io.BytesIO(raw))
                print(f"[DEBUG] Loaded {file.filename} with encoding {enc}")
                break
            except UnicodeDecodeError:
                continue
        if df is None or df.empty:
            return {"error": "Failed to read file or file is empty."}
    except Exception as e:
        return {"error": f"File read failed: {e}"}
    # ======================================================================================

    
    

    # AI ANALYSIS SECTION
    # ==================================================================================================
    try:
        ai_analysis = analyze_larger_dataframe(df)

        # Checks if the ai analysis is an instance of a string, it loads it as a JSON(that's how we want it) if it is
        if isinstance(ai_analysis, str):        
            ai_analysis = json.loads(ai_analysis)

        #  If ai_analysis is neither a string nor a dictionary, it's considered to be in an invalid format.
        elif not isinstance(ai_analysis, dict): 
            ai_analysis = {"error": "Invalid AI result format"}

        # After the checks and proper conversion if the ai analysis, it is passed to gemini_result variable
        gemini_result = ai_analysis

    
    except ZeroDivisionError: # This error is so subtle and caused a lot if issues
        gemini_result = {"error": "AI analysis failed: division by zero"}
    except Exception as e:
        gemini_result = {"error": f"AI analysis failed: {e}"}
        # ==================================================================================================



    # DATA INFORMATION SECTION
    # =======================================================================================================
    # DATA OVERVIEW ERROR HANDLING
    try:
        overview = get_data_overview(df)
    except Exception as e:
        overview = {"error": f"Overview failed: {e}"}

    # Missing‑data ERROR HANDLING
    try:
        missing_data = analyze_missing_data(df)
    except Exception as e:
        missing_data = {"error": f"Missing-data analysis failed: {e}"}

    # =================================================================================================


    # Return everything (now guaranteed to exist)
    return {
        "overview": overview,
        "missing_data": missing_data,
        "gemini_result": gemini_result
    }
