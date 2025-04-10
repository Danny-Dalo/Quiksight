from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from fastapi import UploadFile, File, HTTPException
import os

ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

router = APIRouter()
templates = Jinja2Templates(directory="app_quiksight/templates")




@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request : Request, file : UploadFile = File(...)):
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    # TODO : Handle saving/cleaning here

    return templates.TemplateResponse("home.html", {"request" : request})
 
