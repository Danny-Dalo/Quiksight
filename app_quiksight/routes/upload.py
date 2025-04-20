import os
from fastapi import UploadFile, File
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from app_quiksight.services.analyze import analyze_data
from fastapi.responses import HTMLResponse, RedirectResponse

ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

router = APIRouter()

templates = Jinja2Templates(directory="app_quiksight/templates")


# by default uploaded file is none as no one has uploaded anything
uploaded_file = None 


@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    global uploaded_file


    # ===============================================================================
    if not file or file.filename == "":  # returns error message if file input is empty
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": "Please upload a file before submitting."
        }) 
    # ===============================================================================

    
    # ========================================================================
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} extensions allowed."
        })  # error message if file format is not supported
    # =========================================================================
    


    # =========================================================================

# after a proper file format has been accepted and confirmed
    uploaded_file = file
    global analysis_result
    
    # # It pauses the current function and executes the analyze_data function before returning the RedirectResponse
    # await keyword ensures that the upload_file function will remain paused until the analyze_data co-routine finishes executing and returns a result
    analysis_result = await analyze_data(uploaded_file)
    

    # it unpauses and redirects the function after the await function has returned a value
    return RedirectResponse(url="/api/results", status_code=303)
    # =========================================================================





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
        "analysis": analysis_result
    })




