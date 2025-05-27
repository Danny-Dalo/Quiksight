from dotenv import load_dotenv
import os
from pathlib import Path

# Get the root directory of the project
ROOT_DIR = Path(__file__).parent.parent

# Load environment variables from .env file in the root directory
load_dotenv(dotenv_path=ROOT_DIR / '.env')

# Get the API key
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = "AIzaSyB3Ft4l3D1qyQDXpcLjEE5QdKHNXF4o3Zc"
print(f"API KEY LOADED")


if not GEMINI_API_KEY:
    raise ValueError("GEMINI API KEY is not set in the .env file. Please add your API key to the .env file.")

