
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

# Get the API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("=======================LOADED==========================")

if not GEMINI_API_KEY:
    logger.error("Error with API key")
    raise ValueError(
        "GEMINI_API_KEY is not set in the .env file. "
        "Please add your API key to the .env file."
    )