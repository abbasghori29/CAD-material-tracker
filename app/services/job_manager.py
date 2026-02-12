"""
Job manager - Functions for creating, retrieving, and managing processing jobs.
"""

import asyncio
import uuid
from typing import Dict, Optional

from app.models.job import JobState, JobStatus
from app.services.cleanup_service import cleanup_job_resources

# Global job registry
ACTIVE_JOBS: Dict[str, JobState] = {}
JOB_TASKS: Dict[str, asyncio.Task] = {}
CANCELLATION_TASKS: Dict[str, asyncio.Task] = {}  # Track delayed cancellation tasks


def create_job(pdf_path: str, start_page: int, end_page: int) -> JobState:
    """Create a new job"""
    job_id = str(uuid.uuid4())[:8]  # Short UUID
    job = JobState(job_id, pdf_path, start_page, end_page)
    ACTIVE_JOBS[job_id] = job
    print(f"[JOB-{job_id}] Created: {pdf_path}, pages {start_page}-{end_page}")
    return job


def get_job(job_id: str) -> Optional[JobState]:
    """Get job by ID"""
    return ACTIVE_JOBS.get(job_id)


def cleanup_job(job_id: str):
    """Clean up completed job after some time"""
    if job_id in ACTIVE_JOBS:
        del ACTIVE_JOBS[job_id]
    if job_id in JOB_TASKS:
        del JOB_TASKS[job_id]
    if job_id in CANCELLATION_TASKS:
        del CANCELLATION_TASKS[job_id]
    print(f"[JOB-{job_id}] Cleaned up")


async def schedule_job_cancellation(job_id: str, delay: int = 600):
    """Schedule job cancellation after delay if no reconnection (default: 10 minutes)"""
    await asyncio.sleep(delay)
    
    job = get_job(job_id)
    if job and len(job.subscribers) == 0 and job.status == JobStatus.RUNNING:
        print(f"[JOB-{job_id}] ⏰ No reconnection after {delay}s - cancelling job")
        job.status = JobStatus.FAILED
        job.error = "Cancelled - no active clients for 60 seconds"
        cleanup_job_resources(job_id)  # Clean up saved images and PDF
        cleanup_job(job_id)
    elif job_id in CANCELLATION_TASKS:
        # Job was reconnected - remove cancellation task
        del CANCELLATION_TASKS[job_id]


def get_all_jobs() -> Dict[str, JobState]:
    """Get all active jobs"""
    return ACTIVE_JOBS


def register_job_task(job_id: str, task: asyncio.Task):
    """Register an asyncio task for a job"""
    JOB_TASKS[job_id] = task


def register_cancellation_task(job_id: str, task: asyncio.Task):
    """Register a cancellation task for a job"""
    CANCELLATION_TASKS[job_id] = task


def cancel_cancellation_task(job_id: str):
    """Cancel a pending cancellation task (e.g., on reconnection)"""
    if job_id in CANCELLATION_TASKS:
        task = CANCELLATION_TASKS[job_id]
        if not task.done():
            task.cancel()
        del CANCELLATION_TASKS[job_id]
        print(f"[JOB-{job_id}] Cancellation task cancelled due to reconnection")
