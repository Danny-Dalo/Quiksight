from fastapi import FastAPI, Request
from .routes import upload
from .services import analyze
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse





app = FastAPI()
app.mount("/static", StaticFiles(directory="app_quiksight/static"), name="static")




templates = Jinja2Templates("app_quiksight/templates")
@app.get("/", response_class=HTMLResponse, tags=["Home Page"], description="Gets the home page of the application")

async def home(request : Request):
    # returns the home.html page
    return templates.TemplateResponse("home.html", {"request" : request})




# include the upload API router
app.include_router(upload.router, prefix="/api", tags=["upload"])

# include the analyze API router
app.include_router(analyze.router, prefix="/api", tags=["analyze"])

