from fastapi import FastAPI, Request
from .routes import upload, chat
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse



templates = Jinja2Templates("app_quiksight/templates")

app = FastAPI()


app.mount("/static", StaticFiles(directory="app_quiksight/static"), name="static")



@app.get("/", response_class=HTMLResponse)
async def home(request : Request):
    return templates.TemplateResponse("home.html", {"request" : request})

# ============================================================
# Render hosting service spins down after a while
# So we send a get request every 10 mins to make sure the site is always up and running
# Users don't have to wait for the render load up time
@app.get("/ping")
async def ping():
    return {"status": "alive"}

from fastapi import Response

@app.head("/ping")
async def head_ping():
    return Response(status_code=200)

# =====================================================


app.include_router(upload.router, tags=["upload"])


app.include_router(chat.router, tags=["chat"])


# run command: uv run uvicorn app_quiksight.main:app --reload
