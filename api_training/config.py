from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env into the environment

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

