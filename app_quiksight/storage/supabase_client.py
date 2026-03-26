import os
import logging
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

logger = logging.getLogger("SUPABASE")

url: str = os.getenv("SUPABASE_URL", "")
key: str = os.getenv("SUPABASE_ANON_KEY", "")

if not url or not key:
    logger.warning("SUPABASE_URL and SUPABASE_ANON_KEY environment variables are missing! Authentication will fail.")

supabase: Client = create_client(url, key)
