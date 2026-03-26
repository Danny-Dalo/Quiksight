from fastapi import APIRouter, Request, Form, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import logging
from app_quiksight.storage.supabase_client import supabase

router = APIRouter(tags=["auth"])
templates = Jinja2Templates(directory="app_quiksight/templates")
logger = logging.getLogger("AUTH")

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # If user is already logged in, redirect home
    if request.cookies.get("access_token"):
        try:
            if supabase.auth.get_user(request.cookies.get("access_token")):
                return RedirectResponse(url="/", status_code=303)
        except Exception:
            pass
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    if request.cookies.get("access_token"):
        try:
            if supabase.auth.get_user(request.cookies.get("access_token")):
                return RedirectResponse(url="/", status_code=303)
        except Exception:
            pass
    return templates.TemplateResponse("login.html", {"request": request, "signup": True})

@router.get("/auth/google")
async def google_login(request: Request):
    try:
        redirect_url = str(request.base_url).rstrip("/") + "/auth/callback"
        response = supabase.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                "redirect_to": redirect_url
            }
        })
        return RedirectResponse(url=response.url, status_code=303)
    except Exception as e:
        logger.error(f"Google login initiation failed: {e}")
        return RedirectResponse(url="/login?error=init_failed", status_code=303)

@router.get("/auth/callback")
async def auth_callback(request: Request, code: str = None, next: str = "/"):
    if code:
        try:
            response = supabase.auth.exchange_code_for_session({"auth_code": code})
            red = RedirectResponse(url=next if next.startswith("/") else "/", status_code=303)
            red.set_cookie("access_token", response.session.access_token, httponly=True, max_age=3600*24*7)
            red.set_cookie("refresh_token", response.session.refresh_token, httponly=True, max_age=3600*24*7)
            return red
        except Exception as e:
            logger.error(f"Failed to exchange code: {e}")
            return RedirectResponse(url="/login?error=auth_failed", status_code=303)
            
    return RedirectResponse(url="/login?error=missing_code", status_code=303)

@router.get("/logout")
async def logout():
    try:
        supabase.auth.sign_out()
    except Exception as e:
        pass
    red = RedirectResponse(url="/", status_code=303)
    red.delete_cookie("access_token")
    red.delete_cookie("refresh_token")
    return red
