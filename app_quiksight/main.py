from fastapi import FastAPI, Request, Response, status
from .routes import upload, chat
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import logging

templates = Jinja2Templates("app_quiksight/templates")


from starlette.middleware.base import BaseHTTPMiddleware
MAX_SIZE = 30 * 1024 * 1024  # 30 MB
# Checks file size before it hits upload endpoint logic
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length:
                if int(content_length) > MAX_SIZE:
                    return templates.TemplateResponse(
                        "home.html", 
                        {
                            "request": request, 
                            "error": "File too large. Maximum size limit is 30MB."
                        },
                        status_code=413
                    )
        
        return await call_next(request)


logger = logging.getLogger(__name__)


app = FastAPI()
app.add_middleware(LimitUploadSizeMiddleware)


app.mount("/static", StaticFiles(directory="app_quiksight/static"), name="static")



@app.get("/", response_class=HTMLResponse)
async def home(request : Request):
    logger.info("Page Loaded")
    return templates.TemplateResponse(request=request, name="home.html")

# ============================================================
# Render hosting service spins down after a while
# So we send a get request every 10 mins to make sure the site is always up and running
# Users don't have to wait for the render load up time


@app.get("/ping")
async def ping():
    logger.info("Ping endpoint hit")
    return {"status": "alive"}


@app.head("/ping")
async def head_ping():
    logger.info("Head ping endpoint hit")
    return Response(status_code=200)

# =====================================================


app.include_router(upload.router, tags=["upload"])


app.include_router(chat.router, tags=["chat"])


# run command: uv run uvicorn app_quiksight.main:app --reload
