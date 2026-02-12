"""
Jobs routes - Job status and management endpoints
"""

from fastapi import APIRouter, Depends

from app.api.deps.auth import get_current_user
from app.services.job_manager import get_job, get_all_jobs
from app.models.job import JobStatus

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("/jobs")
async def list_active_jobs():
    """List all active jobs (for reconnection)"""
    jobs = get_all_jobs()
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job.status,
                "current_page": job.current_page,
                "total_pages": job.total_pages,
                "results_count": len(job.results),
                "started_at": job.started_at.isoformat(),
                "subscribers": len(job.subscribers)
            }
            for job_id, job in jobs.items()
        ]
    }


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job"""
    job = get_job(job_id)
    if not job:
        return {"error": "Job not found"}
    
    return {
        "job_id": job_id,
        "status": job.status,
        "current_page": job.current_page,
        "total_pages": job.total_pages,
        "results_count": len(job.results),
        "started_at": job.started_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error,
        "subscribers": len(job.subscribers)
    }


@router.get("/download")
async def download_results():
    """Download results CSV file"""
    import os
    from fastapi.responses import FileResponse
    from app.core.config import RESULTS_FOLDER
    
    csv_path = os.path.join(RESULTS_FOLDER, "results.csv")
    if os.path.exists(csv_path):
        return FileResponse(
            csv_path,
            media_type="text/csv",
            filename="material_results.csv"
        )
    else:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=404,
            content={"error": "Results file not found. Process a PDF first."}
        )
