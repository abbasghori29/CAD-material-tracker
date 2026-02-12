"""
WebSocket endpoint - Real-time communication with clients
"""

import asyncio
import time
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette import status

from app.core.config import UPLOAD_FOLDER
from app.core.security import decode_access_token
from app.models.job import JobStatus
from app.utils.security_utils import validate_pdf_path_for_job
from app.services.job_manager import (
    create_job, 
    get_job, 
    cleanup_job,
    register_job_task,
    register_cancellation_task,
    cancel_cancellation_task,
    schedule_job_cancellation
)
from app.services.cleanup_service import cleanup_job_resources
from app.services.pdf_processor import process_pdf_with_job

router = APIRouter()

# Active WebSocket connections
active_connections: List[WebSocket] = []


async def broadcast(message: dict):
    """Send message to all connected WebSocket clients"""
    if not active_connections:
        print(f"[BROADCAST] No active connections!")
        return
    
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            print(f"[BROADCAST] Send failed: {type(e).__name__}: {e}")
            disconnected.append(connection)
    
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)
            print(f"[BROADCAST] Removed dead connection")


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint with job-based processing.

    Requires a valid JWT access token passed as a `token` query parameter.
    """
    token = websocket.query_params.get("token")
    if not token or decode_access_token(token) is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return

    await websocket.accept()
    active_connections.append(websocket)
    print(f"[WS] Connection accepted")
    
    current_job = None
    heartbeat_task = None
    job_completed_at: float | None = None  # When job finished; used to close after grace period

    async def heartbeat():
        """Send heartbeat only while job is active; stop after completion to save resources"""
        count = 0
        errors = 0
        heartbeat_interval = 1
        dummy_data = "X" * 100

        while True:
            await asyncio.sleep(heartbeat_interval)
            count += 1

            job_active = False
            if current_job:
                job_status = current_job.status
                job_active = job_status == JobStatus.RUNNING
                if job_status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    # Stop sending heartbeats as soon as job is done
                    break
            else:
                # No job yet: send light heartbeat
                pass

            try:
                if job_active:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "n": count,
                        "keep_alive": dummy_data
                    })
                    if count % 30 == 0:
                        await websocket.send_json({
                            "type": "log",
                            "level": "info",
                            "message": f"⚡ Connection alive ({count}s)"
                        })
                elif not current_job:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "n": count,
                        "keep_alive": dummy_data
                    })
            except WebSocketDisconnect:
                break
            except Exception as e:
                errors += 1
                if current_job and current_job.status == JobStatus.RUNNING:
                    print(f"[WS-HEARTBEAT] FAILED #{errors} at {count}s: {type(e).__name__}")
                if errors >= 3:
                    break

    try:
        heartbeat_task = asyncio.create_task(heartbeat())
        print(f"[WS] Heartbeat started")

        # Main message loop
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            # Log only non-ping to avoid flooding logs and memory
            if action != "ping":
                print(f"[WS] Received: {action} (Data: {str(data)[:100]}...)")

            # Track when job completed so we can close after grace period
            if current_job and current_job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                if job_completed_at is None:
                    job_completed_at = time.monotonic()

            # Handle ping (keepalive): respond with pong, then close if job done and grace period passed
            if action == "ping":
                await websocket.send_json({"type": "pong"})
                if job_completed_at is not None and (time.monotonic() - job_completed_at) >= 5:
                    print(f"[WS] Job finished; closing connection after grace period")
                    break
                continue
            
            # Handle process (start new job) - frontend sends 'process'
            if action == "process":
                pdf_path = data.get("pdf_path")
                start_page = int(data.get("start_page", 1))
                end_page = int(data.get("end_page", 1))
                
                if not validate_pdf_path_for_job(pdf_path, UPLOAD_FOLDER):
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid or unauthorized PDF path"
                    })
                    continue
                
                print(f"[WS] Starting job: {pdf_path}, pages {start_page}-{end_page}")
                
                # Create job
                current_job = create_job(pdf_path, start_page, end_page)
                current_job.add_subscriber(websocket)
                
                # Send job ID to client
                await websocket.send_json({
                    "type": "job_created",
                    "job_id": current_job.job_id,
                    "message": f"Job {current_job.job_id} created"
                })
                
                # Start processing in background
                task = asyncio.create_task(process_pdf_with_job(current_job))
                register_job_task(current_job.job_id, task)
                
            # Handle subscribe (reconnect to existing job) - frontend sends 'subscribe'
            elif action == "subscribe":
                job_id = data.get("job_id")
                job = get_job(job_id)
                
                if job:
                    # Cancel any pending cancellation
                    cancel_cancellation_task(job_id)
                    
                    job.add_subscriber(websocket)
                    current_job = job
                    
                    await job.send_summary(websocket)
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Job {job_id} not found"
                    })
            
            elif action == "cancel":
                if current_job:
                    current_job.status = JobStatus.FAILED
                    current_job.error = "Cancelled by user"
                    cleanup_job_resources(current_job.job_id)
                    cleanup_job(current_job.job_id)
                    
                    await websocket.send_json({
                        "type": "cancelled",
                        "message": "Job cancelled"
                    })
                    current_job = None
                    

    except WebSocketDisconnect as e:
        print(f"[WS] Client disconnected: code={e.code}, reason={e.reason if hasattr(e, 'reason') else 'None'}")
    except Exception as e:
        print(f"[WS] UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up this WebSocket
        if websocket in active_connections:
            active_connections.remove(websocket)
        
        # Unsubscribe from job
        if current_job:
            current_job.remove_subscriber(websocket)
            
            # If no subscribers left, schedule delayed cancellation (allows reconnection)
            if len(current_job.subscribers) == 0 and current_job.status == JobStatus.RUNNING:
                task = asyncio.create_task(schedule_job_cancellation(current_job.job_id, delay=600))
                register_cancellation_task(current_job.job_id, task)
                print(f"[WS] ⏰ Scheduled cancellation for job {current_job.job_id} in 10 minutes (reconnect to resume)")
            else:
                print(f"[WS] Unsubscribed from job {current_job.job_id} ({len(current_job.subscribers)} subscribers remaining)")
        
        # Stop heartbeat
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except:
                pass
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print(f"[WS] Connection closed and cleaned up")
