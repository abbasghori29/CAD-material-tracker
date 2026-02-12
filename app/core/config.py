"""
Configuration settings for the CAD Material Tracker application.
Centralizes all environment variables and application settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# === FOLDER CONFIGURATION ===
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output_images"
RESULTS_FOLDER = "results"
IMAGES_FOLDER = "output_images"

# Ensure folders exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, RESULTS_FOLDER, "static", "templates", IMAGES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# === DATABASE CONFIGURATION ===
DATABASE_URL = os.getenv("DATABASE_URL", "")
DIRECT_DATABASE_URL = os.getenv("DIRECT_URL", DATABASE_URL)


# === AUTH / SECURITY CONFIGURATION ===
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret-in-prod")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60")
)

# Comma-separated list of allowed CORS origins (production). If set, overrides dev defaults.
ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "").strip()

# === API KEYS ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "cad-drawing-iy9tc/11")
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY", "")

# === FEATURE FLAGS ===
AUTO_CLEANUP = os.getenv("AUTO_CLEANUP", "false").lower() == "true"
ALWAYS_PYMUPDF = os.getenv("ALWAYS_PYMUPDF", "True").lower() in ("true", "1", "yes")

# === SERVICE AVAILABILITY FLAGS ===
# These will be set by the service modules upon initialization
OPENAI_AVAILABLE = bool(OPENAI_API_KEY)
ROBOFLOW_AVAILABLE = False  # Set by ai_service
PYMUPDF_AVAILABLE = False   # Set by text_extractor
TESSERACT_AVAILABLE = False # Set by ocr_service

# === TAG DESCRIPTIONS ===
# Global storage for uploaded tag descriptions
TAG_DESCRIPTIONS = {}

def load_tag_descriptions():
    """Load material descriptions from JSON file if exists"""
    global TAG_DESCRIPTIONS
    import json
    if os.path.exists("material_descriptions.json"):
        with open("material_descriptions.json", "r", encoding="utf-8") as f:
            TAG_DESCRIPTIONS = json.load(f)
    return TAG_DESCRIPTIONS

# Load on import
load_tag_descriptions()
