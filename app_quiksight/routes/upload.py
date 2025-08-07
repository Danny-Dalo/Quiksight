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
uploaded_dataframe = None


# /chat/upload ==============WHAT HAPPENS WHEN A USER CLICKS THE SUBMIT BUTTON AFTER A FILE HAS BEEN UPLOADED
@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    # UploadFile: A class used by FastAPI to handle uploaded files. It's a file-like object that gives you access to the file's contents, filename, and other metadata.
    # file: UploadFile = File(...): This is the core of the file upload mechanism.
    # The parameter is named file.
    # Its type hint is UploadFile, which tells FastAPI to expect an uploaded file.
    # The File(...) part is the key: it instructs FastAPI to look for a file in the incoming request body and inject it into this function as an UploadFile object.
    # They need to come together
    # the type hint defines the expected data structure, and the dependency injector defines how to retrieve that data from the incoming request
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
    # matches.csv returns (matches, csv)
    if ext not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} extensions allowed."
        })  # error message if file format is not supported
    # =========================================================================
    


    # =========================================================================

    # after a proper file format has been accepted and confirmed, store the file that has been uploaded into the uploaded_file variable
    uploaded_file = file
    
    


    # ========================================================================================
    global analysis_result
    analysis_result = await analyze_data(file)  # The return values(s) of the analyze_data() function, which takes in our uploaded file, is stored in analysis_result
    df = analysis_result.pop("df", None)
# =========================================================
    global uploaded_dataframe
    uploaded_dataframe = df
# ========================================================================================
    

    # Returns a redirect response to the next Uniform Resource Locator(URL) enpoint
    return RedirectResponse(url="/chat/results", status_code=303)
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



# Get the current state of the uploaded dataframe
def get_uploaded_dataframe():
    return uploaded_dataframe
