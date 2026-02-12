"""
Cleanup routes - File cleanup and download endpoints
"""

import os
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.api.deps.auth import get_current_user
from app.core.config import RESULTS_FOLDER
from app.services.cleanup_service import cleanup_temp_files

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("/download-csv")
async def download_csv():
    """Download results CSV"""
    csv_path = os.path.join(RESULTS_FOLDER, "results.csv")
    if os.path.exists(csv_path):
        return FileResponse(csv_path, filename="results.csv", media_type="text/csv")
    return {"error": "No results file found"}


@router.post("/cleanup")
async def cleanup_endpoint():
    """Manually trigger cleanup of temporary files"""
    cleaned = cleanup_temp_files()
    return {
        "success": True,
        "message": f"Cleaned up {cleaned} temporary files"
    }
