"""
Cleanup service - Functions for cleaning up temporary files and resources.
"""

import os
import asyncio
import shutil
import time
from datetime import datetime
from typing import Optional

from sqlalchemy import text

from app.core.config import UPLOAD_FOLDER, RESULTS_FOLDER, IMAGES_FOLDER
from app.services.image_utils import cleanup_job_images


def ping_database():
    """
    Run a trivial DB query to keep Supabase active (avoids 7-day pause on free tier).
    Called from the daily cleanup job so no separate scheduler is needed.
    """
    try:
        from app.db.session import SessionLocal
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            db.commit()
            print("[CLEANUP] Database ping OK (Supabase keep-alive)")
        finally:
            db.close()
    except Exception as e:
        print(f"[CLEANUP] Database ping failed (non-fatal): {e}")


def cleanup_job_pdf(job_id: str, pdf_path: Optional[str] = None, max_retries: int = 5, retry_delay: float = 2.0):
    """Delete the uploaded PDF file for a job with retry logic"""
    # Import here to avoid circular imports
    from app.services.job_manager import get_job
    
    if pdf_path is None:
        job = get_job(job_id)
        if job and job.pdf_path:
            pdf_path = job.pdf_path
        else:
            return
    
    if not os.path.exists(pdf_path):
        return  # File already deleted
    
    # Retry deletion if file is in use
    for attempt in range(max_retries):
        try:
            # Try to delete the file
            os.remove(pdf_path)
            print(f"[JOB-{job_id}] Deleted PDF: {pdf_path}")
            return  # Success
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"[JOB-{job_id}] PDF in use, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                print(f"[JOB-{job_id}] Error deleting PDF {pdf_path} after {max_retries} attempts: {e}")
                # Schedule deletion for later
                schedule_delayed_pdf_deletion(pdf_path)
        except Exception as e:
            print(f"[JOB-{job_id}] Error deleting PDF {pdf_path}: {e}")
            break


def schedule_delayed_pdf_deletion(pdf_path: str, delay: int = 300):
    """Schedule PDF deletion after a delay (for files still in use)"""
    async def delayed_delete():
        await asyncio.sleep(delay)
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"[CLEANUP] Deleted PDF after delay: {pdf_path}")
        except Exception as e:
            print(f"[CLEANUP] Error deleting PDF {pdf_path} after delay: {e}")
    
    # Schedule the delayed deletion
    asyncio.create_task(delayed_delete())


def cleanup_job_resources(job_id: str):
    """Delete all resources for a job (images + PDF)"""
    cleanup_job_images(job_id)
    cleanup_job_pdf(job_id)


def cleanup_temp_files():
    """Clean up temporary files (PDFs and images) but keep CSV results"""
    cleaned = 0
    
    # Clean uploads folder
    if os.path.exists(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            try:
                filepath = os.path.join(UPLOAD_FOLDER, f)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    cleaned += 1
            except:
                pass
    
    # Clean images folder
    if os.path.exists(IMAGES_FOLDER):
        for f in os.listdir(IMAGES_FOLDER):
            try:
                filepath = os.path.join(IMAGES_FOLDER, f)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    cleaned += 1
            except:
                pass
    
    return cleaned


def scheduled_cleanup_uploads():
    """Clean up uploads, static/images, and results folders daily at 9:17 PM"""
    now = datetime.now()
    print(f"[CLEANUP] Scheduled cleanup started at {now}")

    # Keep Supabase active (free tier pauses after ~7 days inactivity)
    ping_database()
    
    folders_to_clean = [UPLOAD_FOLDER, IMAGES_FOLDER, RESULTS_FOLDER]
    total_deleted = 0
    total_errors = 0
    
    for folder in folders_to_clean:
        if not os.path.exists(folder):
            print(f"[CLEANUP] Folder does not exist: {folder}")
            continue
            
        folder_deleted = 0
        folder_errors = 0
        
        try:
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        folder_deleted += 1
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        folder_deleted += 1
                except PermissionError as e:
                    print(f"[CLEANUP] Permission error deleting {item_path}: {e}")
                    folder_errors += 1
                except Exception as e:
                    print(f"[CLEANUP] Error deleting {item_path}: {e}")
                    folder_errors += 1
        except Exception as e:
            print(f"[CLEANUP] Error listing folder {folder}: {e}")
            folder_errors += 1
        
        print(f"[CLEANUP] Cleaned {folder}: {folder_deleted} items deleted, {folder_errors} errors")
        total_deleted += folder_deleted
        total_errors += folder_errors
    
    print(f"[CLEANUP] Scheduled cleanup completed: {total_deleted} items deleted, {total_errors} errors")
    return {"deleted": total_deleted, "errors": total_errors}


# APScheduler integration
_scheduler = None


def start_scheduled_cleanup():
    """Start the scheduled cleanup task"""
    global _scheduler
    
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        
        _scheduler = AsyncIOScheduler()
        
        # Schedule daily at 9:17 PM
        trigger = CronTrigger(hour=21, minute=17)
        _scheduler.add_job(scheduled_cleanup_uploads, trigger, id="daily_cleanup")
        
        # APScheduler will start when the event loop is running (called from FastAPI startup)
        try:
            _scheduler.start()
            print("[SCHEDULER] APScheduler started - cleanup scheduled at 9:17 PM daily")
            return True
        except RuntimeError as e:
            # No event loop yet - will be started later
            print(f"[SCHEDULER] APScheduler deferred start: {e}")
            return True
        
    except ImportError:
        print("[SCHEDULER] APScheduler not installed - using simple scheduler fallback")
        _start_simple_scheduler()
        return False
    except Exception as e:
        print(f"[SCHEDULER] Error starting APScheduler: {e}")
        _start_simple_scheduler()
        return False


def _start_simple_scheduler():
    """Start the simple fallback scheduler in the proper async context"""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(simple_scheduled_cleanup())
    except RuntimeError:
        # No running loop - schedule to be started later
        print("[SCHEDULER-SIMPLE] Will start when event loop is available")
        pass


async def simple_scheduled_cleanup():
    """Simple asyncio-based scheduler (fallback if APScheduler not available)"""
    while True:
        now = datetime.now()
        # Calculate seconds until 9:17 PM
        target = now.replace(hour=21, minute=17, second=0, microsecond=0)
        if now >= target:
            # Already past 9:17 PM today, schedule for tomorrow
            target = target.replace(day=now.day + 1)
        
        wait_seconds = (target - now).total_seconds()
        print(f"[SCHEDULER-SIMPLE] Next cleanup in {wait_seconds / 3600:.1f} hours")
        
        await asyncio.sleep(wait_seconds)
        scheduled_cleanup_uploads()
