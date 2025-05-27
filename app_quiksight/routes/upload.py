import os
from fastapi import UploadFile, File
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from app_quiksight.services.analyze import analyze_data, run_ai_analysis
from fastapi.responses import HTMLResponse, RedirectResponse

ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

router = APIRouter()

templates = Jinja2Templates(directory="app_quiksight/templates")


# by default uploaded file is none as no one has uploaded anything
uploaded_file = None 




# The /upload endpoint
# Handles file uploads from the user.
# Validates file type. If the file is valid, it assigns it to the uploaded_file variable.
# Calls analyze_data (from analyze.py) for basic analysis of the file
# Calls run_ai_analysis (from analyze.py) for AI analysis of the file
# Redirects to the results page after the analysis is complete.

# will respond to file upload requests with a html response
@router.post("/upload", response_class=HTMLResponse, 
                    description='''
                    # Handles file uploads from the user.
                    # Validates file type. If the file is valid, it assigns it to the uploaded_file variable.
                    # Calls analyze_data (from analyze.py) for basic analysis of the file
                    # Calls run_ai_analysis (from analyze.py) for AI analysis of the file
                    # Redirects to the results page after the analysis is complete.
                    ''') 
# The 'file' parameter receives the uploaded file from the user's form submission.
# It is required (File(...)) and is of type UploadFile, which allows FastAPI to access the file's contents
async def upload_file(request: Request, file: UploadFile = File(...)):
    # we set the global variable uploaded_file to be the file that the user uploaded
    global uploaded_file

    # FILE VALIDATION
    # if the file is not uploaded or the file name is empty, an error message is returned
    if not file or file.filename == "":  
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": "Please upload a file before submitting."
        }) 

    
    # splits the file name(myfile.csv) into the file name(myfile) and the extension(csv)
    _, ext = os.path.splitext(file.filename.lower())

    # FILE VALIDATION
    # if the extension is not in the list of allowed extensions, an error message is returned
    if ext not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} extensions allowed."
        })  
    



    # the uploaded file is assigned to the uploaded_file variable after the file is validated
    uploaded_file = file
    global analysis_result, ai_analysis
    
    # ANALYSIS
    # It pauses the current function and executes the analyze_data function and returns the result before returning the RedirectResponse
    analysis_result = await analyze_data(uploaded_file)
    # process continues in the analyze.py file

    # ANALYSIS
    # It pauses the current function and executes the run_ai_analysis function before returning the RedirectResponse
    ai_analysis = await run_ai_analysis(uploaded_file)

        


    # ========================================================================================
    df = analysis_result.pop("df", None)
    ai_analysis = await run_ai_analysis(df) if df is not None else {"error": "No dataframe for AI analysis"}

    # ========================================================================================
    

    # it unpauses and redirects the function after the await function has returned a value
    return RedirectResponse(url="/api/results", status_code=303)
#     # =========================================================================
















@router.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    global uploaded_file
    
    if not uploaded_file:
        return RedirectResponse(url="/", status_code=303) # ensures that the only way to go to the next page is by submitting an uploaded file
    
    

    # ==============================================================================================
    if "overview" not in analysis_result:
        # if for some reason or error "overview" dict is not returned, it stays at the home page and displays an error message
        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "error": analysis_result.get("error", "Unknown analysis error"),
            },
        )
    if "ai_result" not in ai_analysis:
        # if for some reason or error "overview" dict is not returned, it stays at the home page and displays an error message
        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "error": ai_analysis.get("error", "ai result is not found bro"),
            },
        )
        # ==============================================================================================


    # it then returns the results.html page
    """
    file represents the uploaded file
    request is  simply sending a get request to the server
    analysis is the holy grail analyze_data function doing the heavy processing and lifting
    """
    return templates.TemplateResponse("results.html", {
        "request": request, 
        "file": uploaded_file,
        "analysis": analysis_result,
        "ai_analysis" : ai_analysis
    })


