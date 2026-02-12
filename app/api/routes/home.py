"""
Home route - Serves the main page
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.config import TAG_DESCRIPTIONS
from app.services.ai_service import is_roboflow_available, is_openai_available

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page"""
    response = templates.TemplateResponse("index.html", {
        "request": request,
        "roboflow_available": is_roboflow_available(),
        "openai_available": is_openai_available(),
        "tag_count": len(TAG_DESCRIPTIONS)
    })
    # Prevent browser caching to avoid connection issues on refresh
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response
