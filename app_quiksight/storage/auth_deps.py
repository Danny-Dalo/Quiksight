from fastapi import Request, Response
from app_quiksight.storage.supabase_client import supabase
import logging

logger = logging.getLogger("AUTH_DEPS")

async def get_optional_user(request: Request, response: Response):
    access_token = request.cookies.get("access_token")
    refresh_token = request.cookies.get("refresh_token")
    
    if not access_token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            access_token = auth_header.split(" ")[1]
            
    if not access_token and not refresh_token:
        return None
        
    # fetching user based existing access token
    if access_token:
        try:
            user_response = supabase.auth.get_user(access_token)
            if user_response and user_response.user:
                return user_response.user
        except Exception as e:
            logger.warning(f"Access token invalid or expired: {e}")
    
    # Refreshing token if access token failed or wasn't present
    if refresh_token:
        try:
            # Re-establish the session using the refresh token
            session_response = supabase.auth.set_session(refresh_token)
            if session_response and session_response.session:
                # Updating cookies with new tokens
                response.set_cookie("access_token", session_response.session.access_token, httponly=True, max_age=3600*24*7)
                response.set_cookie("refresh_token", session_response.session.refresh_token, httponly=True, max_age=3600*24*7)
                
                # Fetch user again to ensure validity
                user_response = supabase.auth.get_user(session_response.session.access_token)
                if user_response and user_response.user:
                    return user_response.user
        except Exception as e:
            logger.warning(f"Failed to refresh session: {e}")
            
    return None
