"""
Job models - Status enum and JobState class for managing processing jobs.
"""

from enum import Enum
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import WebSocket


class JobStatus(str, Enum):
    """Job status enum"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobState:
    """Represents a processing job that runs independently of WebSocket"""
    
    def __init__(self, job_id: str, pdf_path: str, start_page: int, end_page: int):
        self.job_id = job_id
        self.pdf_path = pdf_path
        self.start_page = start_page
        self.end_page = end_page
        self.status = JobStatus.QUEUED
        self.current_page = start_page
        self.total_pages = end_page - start_page + 1
        self.results = []
        self.error = None
        self.started_at = datetime.now()
        self.completed_at = None
        self.subscribers: List[WebSocket] = []  # WebSockets watching this job
        self.messages = []  # Store all messages for late joiners
        # Track processed pages with their images for reconnection replay
        self.processed_pages: Dict[int, dict] = {}  # page_num -> {full_page_url, drawings: [...], sheet}
        self.current_drawing_index = 0  # Current drawing being processed
        
    def add_subscriber(self, websocket: WebSocket):
        """Add a WebSocket to watch this job"""
        if websocket not in self.subscribers:
            self.subscribers.append(websocket)
            print(f"[JOB-{self.job_id}] Added subscriber. Total: {len(self.subscribers)}")
    
    def remove_subscriber(self, websocket: WebSocket):
        """Remove a WebSocket from this job"""
        if websocket in self.subscribers:
            self.subscribers.remove(websocket)
            print(f"[JOB-{self.job_id}] Removed subscriber. Total: {len(self.subscribers)}")
            
            # If no subscribers left, schedule delayed cancellation (allow reconnection)
            if len(self.subscribers) == 0 and self.status == JobStatus.RUNNING:
                print(f"[JOB-{self.job_id}] ⚠️ No subscribers left - will cancel in 60s if no reconnection")
                # Don't cancel immediately - allow time for reconnection
                return False  # Don't cancel immediately
        return False
    
    async def broadcast(self, message: dict):
        """Broadcast message to all subscribers - optimized to stop logging after completion"""
        if len(self.subscribers) == 0:
            # Silently skip if no subscribers (job might be cancelled)
            return
        
        # Broadcast to all connected clients
        dead_sockets = []
        msg_type = message.get('type', 'unknown')
        
        # Only log if job is still running (stops logging after completion)
        job_running = self.status == JobStatus.RUNNING
        
        # Only log non-image messages AND only if job is running
        if job_running and msg_type not in ['full_page', 'drawing', 'heartbeat']:
            print(f"[JOB-{self.job_id}] Broadcasting '{msg_type}' to {len(self.subscribers)} subscriber(s)")
        
        for ws in self.subscribers[:]:  # Copy list to avoid modification during iteration
            try:
                await ws.send_json(message)
                # Only log success for important messages AND only if job is running
                if job_running and msg_type not in ['full_page', 'drawing', 'heartbeat']:
                    print(f"[JOB-{self.job_id}] ✅ Sent '{msg_type}' successfully")
            except Exception as e:
                # Only log errors if job is running (reduces I/O after completion)
                if job_running:
                    if 'WebSocketDisconnect' in str(type(e).__name__) or 'ConnectionClosed' in str(type(e).__name__):
                        print(f"[JOB-{self.job_id}] Client disconnected during '{msg_type}' send")
                    else:
                        print(f"[JOB-{self.job_id}] ❌ Failed to send '{msg_type}': {type(e).__name__}: {e}")
                dead_sockets.append(ws)
        
        # Remove dead sockets
        for ws in dead_sockets:
            self.remove_subscriber(ws)
    
    async def send_summary(self, websocket: WebSocket):
        """Send job summary to reconnecting client - WITH IMAGE REPLAY"""
        print(f"[JOB-{self.job_id}] Sending summary with image replay to new subscriber")
        
        # Count what we've done
        pages_processed = self.current_page - self.start_page
        
        try:
            # Send reconnect summary first
            await websocket.send_json({
                "type": "reconnect_summary",
                "job_id": self.job_id,
                "status": self.status,
                "current_page": self.current_page,
                "total_pages": self.total_pages,
                "pages_processed": pages_processed,
                "results_count": len(self.results),
                "message": f"Reconnected! Replaying current page..."
            })
            
            # Send current stats
            await websocket.send_json({
                "type": "log",
                "level": "success",
                "message": f"📊 Reconnected: Page {self.current_page}/{self.end_page} | {len(self.results)} tags found so far"
            })
            
            # Replay current page images if we have them
            if self.current_page in self.processed_pages:
                page_data = self.processed_pages[self.current_page]
                
                # Send full page image
                if page_data.get('full_page_url'):
                    await websocket.send_json({
                        "type": "full_page",
                        "image_url": page_data['full_page_url'],
                        "page": self.current_page,
                        "sheet": page_data.get('sheet', 'N/A'),
                        "drawing_count": len(page_data.get('drawings', []))
                    })
                    print(f"[JOB-{self.job_id}] Replayed full page image for page {self.current_page}")
                
                # Send all drawing images that have been processed
                for drawing in page_data.get('drawings', []):
                    await websocket.send_json({
                        "type": "drawing",
                        "index": drawing['index'],
                        "total_drawings": page_data.get('total_drawings', len(page_data.get('drawings', []))),
                        "image_url": drawing['image_url'],
                        "confidence": drawing.get('confidence', 'N/A'),
                        "bbox": drawing.get('bbox', [])
                    })
                print(f"[JOB-{self.job_id}] Replayed {len(page_data.get('drawings', []))} drawing images")
            
            # Send log about where we are
            await websocket.send_json({
                "type": "log",
                "level": "info",
                "message": f"▶️ Continuing from drawing {self.current_drawing_index + 1} on page {self.current_page}..."
            })
            
            print(f"[JOB-{self.job_id}] Summary sent, will continue from page {self.current_page}, drawing {self.current_drawing_index + 1}")
        except Exception as e:
            print(f"[JOB-{self.job_id}] Failed to send summary: {e}")
