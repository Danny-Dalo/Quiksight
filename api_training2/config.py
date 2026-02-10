
from dotenv import load_dotenv
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent

# Load environment variables from .env file in the root directory
load_dotenv(dotenv_path=ROOT_DIR / '.env')

# Get the API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview")

print("=======================LOADED==========================")

if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY is not set. OpenRouter features will not work.")
    
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set. Gemini features will not work.")

# Require at least one API key
if not OPENROUTER_API_KEY and not GEMINI_API_KEY:
    raise ValueError(
        "No API keys configured. Please set either OPENROUTER_API_KEY or GEMINI_API_KEY in your .env file."
    )