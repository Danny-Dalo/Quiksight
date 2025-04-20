from fastapi import FastAPI, Request
from .routes import upload
from .services import analyze
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse



templates = Jinja2Templates("app_quiksight/templates")

app = FastAPI()
app.mount("/static", StaticFiles(directory="app_quiksight/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request : Request):
    return templates.TemplateResponse("home.html", {"request" : request})


app.include_router(upload.router, prefix="/api", tags=["upload"])

app.include_router(analyze.router, prefix="/api", tags=["analyze"])

