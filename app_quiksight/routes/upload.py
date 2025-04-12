from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

from fastapi import UploadFile, File, HTTPException
import os
from app_quiksight.services.analyze import analyze_data

ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

router = APIRouter()
templates = Jinja2Templates(directory="app_quiksight/templates")




uploaded_file = None


@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    global uploaded_file


    # ===============================================================================
    if not file or file.filename == "":
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": "Please upload a file before submitting."
        })
    # ===============================================================================

    
    
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} allowed."
        })
    
    uploaded_file = file
    global analysis_result
    analysis_result = await analyze_data(uploaded_file)
    
    
    
    return RedirectResponse(url="/api/results", status_code=303)




@router.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    global uploaded_file
    
    if not uploaded_file:
        return RedirectResponse(url="/", status_code=303)
    
    
    
    return templates.TemplateResponse("results.html", {
        "request": request, 
        "file": uploaded_file,
        "analysis": analysis_result
    })





