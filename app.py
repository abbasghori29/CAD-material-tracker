"""
CAD Material Tracker - FastAPI Web Application
Real-time UI with WebSockets for live updates
"""

# ============================================
# DISABLE WEBSOCKET PING/PONG - MUST BE FIRST
# This prevents disconnections when browser tab is inactive
# Browser throttles inactive tabs and can't respond to pings
# ============================================
try:
    import websockets.server
    _original_serve = websockets.server.serve
    
    def _serve_no_ping(*args, **kwargs):
        """Monkey-patch to disable WebSocket ping/pong"""
        kwargs['ping_interval'] = None  # Disable server pings
        kwargs['ping_timeout'] = None   # Disable ping timeout
        return _original_serve(*args, **kwargs)
    
    websockets.server.serve = _serve_no_ping
    print("[WS-CONFIG] WebSocket ping/pong DISABLED - tab inactivity won't cause disconnects")
except Exception as e:
    print(f"[WS-CONFIG] Warning: Could not disable WebSocket pings: {e}")

from fastapi import FastAPI, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import base64
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
import gc  # Garbage collection for memory cleanup
import uuid
from enum import Enum

# Enable nested event loops - REQUIRED for LlamaParse which uses async internally
# This must be done BEFORE any event loops are created
try:
    import nest_asyncio
    nest_asyncio.apply()
    print("[INIT] nest_asyncio applied - nested event loops enabled")
except ImportError:
    print("[INIT] WARNING: nest_asyncio not installed - LlamaParse may fail")
    print("[INIT] Install with: pip install nest-asyncio")

# Load environment variables
load_dotenv()

# ========================================
# JOB MANAGEMENT SYSTEM - BULLETPROOF
# ========================================

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

# OpenAI Vision for location description extraction
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

class DrawingLocationInfo(BaseModel):
    """Extracted location information from a CAD drawing."""
    drawing_id: str = Field(
        description="The alphanumeric code/identifier for the drawing, usually found to the left of the title or in a circle. Examples: '1/A101', '5', 'A-2'. Return empty string if not found."
    )
    location_description: str = Field(
        description="The location identifier/description found in the drawing title block or header. Examples: 'E1 1/8 RCP - LEVEL 3 PART B', 'C3 KITCHEN E1 ELEVATION 2', 'B1 SECTION AT EAST OF GARAGE - NS'. Return empty string if not found."
    )

# Shared prompt for location extraction
LOCATION_EXTRACTION_PROMPT = """Extract the drawing identifier and title/location from this CAD drawing image.

Look carefully at these areas:
- The circular bubble or rectangular box usually to the left of or above the main title (contains drawing ID).
- Title block (usually bottom-right corner or top of page).
- Drawing header/label (large text near top).

Extraction Rules:
1. Drawing ID: Extract the alphanumeric code (e.g., '1', '1-A101', 'A', 1 IN210, 2 ID202) that identifies this specific drawing on the sheet. It is often closest to the left of the location text.
2. Location: Extract the main title or location identifier (e.g., 'LEVEL 3 PLAN', 'SECTION AT GARAGE').

Return both fields. Use empty string if a field is not found."""

# OpenAI setup for vision
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_AVAILABLE = bool(OPENAI_API_KEY)
VISION_LLM = None

if OPENAI_AVAILABLE:
    try:
        VISION_LLM = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=OPENAI_API_KEY,
            timeout=60,
            max_retries=2
        )
        # OpenAI Vision enabled with gpt-4o model
    except Exception as e:
        print(f"WARNING: Failed to initialize OpenAI: {e}")
        OPENAI_AVAILABLE = False

# Import extraction logic
import pdfplumber
import re
import csv
from PIL import Image, ImageFile, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import tempfile

# === CONFIG ===
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output_images"
RESULTS_FOLDER = "results"
IMAGES_FOLDER = "static/images"  # Server-side image storage for WebSocket
AUTO_CLEANUP = os.getenv("AUTO_CLEANUP", "false").lower() == "true"  # Set AUTO_CLEANUP=true in .env to enable

# Ensure folders exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, RESULTS_FOLDER, "static", "templates", IMAGES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Prevent PIL crash
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 500_000_000

# Regex for tags - FLEXIBLE to handle any length and common CAD separators
# Common CAD separators: hyphen (-), underscore (_), period (.), slash (/)
# Pattern matches: LETTERS + optional separator + optional more LETTERS + separator + DIGIT-LIKE + optional suffix
# Examples: BR-1, A_WC-12, A_PT-1A, APT-IA (1→I OCR error), WC-12
# Requires at least one digit-like character: 0-9, I, l, O (common OCR confusions)
TAG_PATTERN = re.compile(r'(?i)[A-Z]+[-_./]?[A-Z]*[-_./]?[0-9IlO][A-Z0-9IlO]*')
# Sheet number detection is done in get_sheet_number() function with scoring logic
# Common sheet prefixes: A=Architectural, E=Electrical, S=Structural, M=Mechanical, 
# P=Plumbing, C=Civil, G=General, L=Landscape, D=Details
# Digit-like: 0-9, I (often 1), l (often 1), O (often 0)
# This pattern ensures at least one digit-like char exists in the tag

# Roboflow setup
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "cad-drawing-iy9tc/11")
CLIENT = None
ROBOFLOW_AVAILABLE = False

try:
    import importlib
    inference_sdk = importlib.import_module('inference_sdk')
    InferenceHTTPClient = getattr(inference_sdk, 'InferenceHTTPClient')
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )
    ROBOFLOW_AVAILABLE = True
except ImportError:
    pass

# OCR Setup - Multi-engine with LlamaParse as primary (best accuracy for CAD drawings)
import numpy as np

# LlamaParse - BEST for CAD drawings (most accurate, handles complex layouts)
LLAMA_PARSE = None
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY", "")
try:
    from llama_parse import LlamaParse
    if LLAMA_PARSE_API_KEY:
        LLAMA_PARSE = LlamaParse(api_key=LLAMA_PARSE_API_KEY, result_type="text")
        print("[OCR] LlamaParse initialized - best accuracy for CAD drawings")
    else:
        print("[OCR] LlamaParse API key not set - set LLAMA_PARSE_API_KEY in .env")
except ImportError:
    print("[OCR] LlamaParse not installed - install with: pip install llama-parse")
except Exception as e:
    print(f"[OCR] LlamaParse initialization failed: {e}")

# EasyOCR - Fallback option
EASY_OCR_READER = None
try:
    import easyocr
    EASY_OCR_READER = easyocr.Reader(['en'], gpu=False)
    print("[OCR] EasyOCR initialized as fallback")
except (ImportError, OSError) as e:
    print(f"[OCR] EasyOCR not available: {e}")

# Tesseract - Last resort fallback
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    print("[OCR] Tesseract available as last resort")
except ImportError:
    pass

# FastAPI app
app = FastAPI(title="CAD Material Tracker")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output_images", StaticFiles(directory="output_images"), name="output_images")
templates = Jinja2Templates(directory="templates")

# WebSocket connections
active_connections: List[WebSocket] = []

# Load material descriptions
TAG_DESCRIPTIONS = {}
if os.path.exists("material_descriptions.json"):
    with open("material_descriptions.json", "r", encoding="utf-8") as f:
        TAG_DESCRIPTIONS = json.load(f)


def image_to_base64(img: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    img.save(buffered, format=format, quality=quality)
    return base64.b64encode(buffered.getvalue()).decode()

def save_image_to_disk(img: Image.Image, filename: str, quality: int = 80) -> str:
    """Save image to disk and return URL path"""
    filepath = os.path.join(IMAGES_FOLDER, filename)
    img.save(filepath, "JPEG", quality=quality)
    return f"/static/images/{filename}"

def cleanup_job_images(job_id: str):
    """Delete all images for a job"""
    import glob
    pattern = os.path.join(IMAGES_FOLDER, f"{job_id}-*")
    for filepath in glob.glob(pattern):
        try:
            os.remove(filepath)
        except:
            pass

def cleanup_job_pdf(job_id: str, max_retries: int = 5, retry_delay: float = 2.0):
    """Delete the uploaded PDF file for a job with retry logic"""
    job = get_job(job_id)
    if job and job.pdf_path:
        pdf_path = job.pdf_path
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
                    import time
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


async def broadcast(message: dict):
    """Send message to all connected WebSocket clients"""
    if not active_connections:
        print(f"[BROADCAST] No active connections!")
        return
    
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
            # Force immediate send
            try:
                await connection.flush() if hasattr(connection, 'flush') else None
            except:
                pass
        except Exception as e:
            print(f"[BROADCAST] Send failed: {type(e).__name__}: {e}")
            disconnected.append(connection)
    
    # Clean up disconnected clients
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)
            print(f"[BROADCAST] Removed dead connection")


def detect_drawings_on_page(page, page_num: int):
    """Detect CAD drawings using Roboflow and return full page + cropped drawings"""
    if not ROBOFLOW_AVAILABLE or CLIENT is None:
        return None, [], None
    
    try:
        # Use adaptive resolution based on page size
        page_width, page_height = page.width, page.height
        max_dimension = max(page_width, page_height)
        
        if max_dimension > 1728:  # ~24 inches at 72 dpi
            resolution = 100
        else:
            resolution = 150
        
        img_original = page.to_image(resolution=resolution).original.convert("RGB")
        img_original = ImageOps.exif_transpose(img_original)
        orig_width, orig_height = img_original.size
        
        # Prepare for inference
        img_for_inference = img_original.copy().convert("L").convert("RGB")
        img_resized = img_for_inference.resize((640, 640), Image.Resampling.LANCZOS)
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img_resized.save(temp_file.name, format='JPEG', quality=85)
        temp_file.close()
        
        result = CLIENT.infer(temp_file.name, model_id=ROBOFLOW_MODEL_ID)
        os.unlink(temp_file.name)
        
        scale_x, scale_y = orig_width / 640.0, orig_height / 640.0
        
        drawings = []
        cropped_images = []
        
        if 'predictions' in result:
            # Step 1: Collect all valid detections
            raw_detections = []
            for pred in result['predictions']:
                if pred.get('class') == 'drawing' and pred.get('confidence', 0) > 0.8:
                    x, y = pred['x'] * scale_x, pred['y'] * scale_y
                    w, h = pred['width'] * scale_x, pred['height'] * scale_y
                    x1, y1 = max(0, x - w/2), max(0, y - h/2)
                    x2, y2 = min(orig_width, x + w/2), min(orig_height, y + h/2)
                    
                    if x2 > x1 and y2 > y1:
                        raw_detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "confidence": pred.get('confidence', 0)
                        })

            # Step 2: Apply Non-Maximum Suppression (NMS)
            # Sort by confidence descending
            raw_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            final_detections = []
            while raw_detections:
                current = raw_detections.pop(0)
                final_detections.append(current)
                
                # Compare with remaining detections
                # If overlap is high (IoU > 0.5), discard the lower confidence one
                keep = []
                c_x1, c_y1, c_x2, c_y2 = current['bbox']
                area_current = (c_x2 - c_x1) * (c_y2 - c_y1)
                
                for other in raw_detections:
                    o_x1, o_y1, o_x2, o_y2 = other['bbox']
                    
                    # Calculate intersection
                    ix1 = max(c_x1, o_x1)
                    iy1 = max(c_y1, o_y1)
                    ix2 = min(c_x2, o_x2)
                    iy2 = min(c_y2, o_y2)
                    
                    if ix2 > ix1 and iy2 > iy1:
                        area_inter = (ix2 - ix1) * (iy2 - iy1)
                        area_other = (o_x2 - o_x1) * (o_y2 - o_y1)
                        union = area_current + area_other - area_inter
                        iou = area_inter / union if union > 0 else 0
                        
                        if iou < 0.3:  # Threshold for overlap (30%)
                            keep.append(other)
                    else:
                        keep.append(other)
                
                raw_detections = keep

            # Step 3: Create final objects
            for d in final_detections:
                x1, y1, x2, y2 = d['bbox']
                drawings.append(d)
                cropped = img_original.crop((int(x1), int(y1), int(x2), int(y2)))
                cropped_images.append({
                    "image": cropped,
                    "confidence": d['confidence'],
                    "bbox": (int(x1), int(y1), int(x2), int(y2))
                })
        
        # Draw boxes on full page image (RED color)
        img_annotated = img_original.copy()
        draw = ImageDraw.Draw(img_annotated)
        for i, d in enumerate(drawings):
            x1, y1, x2, y2 = d["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="#FF3B3B", width=4)
            # Draw number label
            draw.rectangle([x1, y1, x1+30, y1+25], fill="#FF3B3B")
            draw.text((x1+8, y1+3), str(i+1), fill="#FFFFFF")
        
        return img_annotated, cropped_images, drawings
        
    except Exception as e:
        print(f"Detection error: {e}")
        return None, [], []


def ocr_image(img: Image.Image) -> str:
    """OCR an image - tries LlamaParse (best), then EasyOCR, then Tesseract"""
    
    # Preprocessing for all OCR engines
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Upscale small images for better accuracy
    width, height = img.size
    if width < 1500 or height < 1500:
        scale_factor = 2
        new_size = (width * scale_factor, height * scale_factor)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Try LlamaParse first (best accuracy for CAD drawings)
    if LLAMA_PARSE is not None:
        temp_path = None
        try:
            # Save image to temp file for LlamaParse
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            img.save(temp_file.name, format='PNG')
            temp_file.close()
            temp_path = temp_file.name
            
            # Get file size for logging
            file_size = os.path.getsize(temp_path)
            print(f"[OCR] LlamaParse: Processing {temp_path} ({file_size} bytes)")
            
            # With nest_asyncio applied at startup, we can use sync load_data directly
            # nest_asyncio allows nested event loops so LlamaParse's internal async works
            extra_info = {"file_name": os.path.basename(temp_path)}
            with open(temp_path, "rb") as f:
                documents = LLAMA_PARSE.load_data(f, extra_info=extra_info)
            
            print(f"[OCR] LlamaParse: Got {len(documents) if documents else 0} document(s)")
            
            if documents:
                text = " ".join([doc.text for doc in documents if doc.text])
                if text.strip():
                    print(f"[OCR] LlamaParse SUCCESS: Extracted {len(text)} chars")
                    return text
                else:
                    print(f"[OCR] LlamaParse: Documents returned but no text extracted")
            else:
                print(f"[OCR] LlamaParse: No documents returned")
        except Exception as e:
            import traceback
            print(f"[OCR] LlamaParse ERROR: {type(e).__name__}: {e}")
            print(f"[OCR] LlamaParse TRACEBACK:\n{traceback.format_exc()}")
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    # Fallback to EasyOCR
    if EASY_OCR_READER is not None:
        try:
            # Convert to grayscale for EasyOCR
            img_gray = img.convert('L')
            img_np = np.array(img_gray)
            results = EASY_OCR_READER.readtext(img_np, detail=0)
            text = " ".join(results)
            if text.strip():
                return text
        except Exception as e:
            print(f"[OCR] EasyOCR error: {e}")
    
    # Last resort: Tesseract
    if TESSERACT_AVAILABLE:
        try:
            from pytesseract import image_to_string
            img_gray = img.convert('L')
            # Enhance contrast for Tesseract
            enhancer = ImageEnhance.Contrast(img_gray)
            img_enhanced = enhancer.enhance(2.0)
            text = image_to_string(img_enhanced)
            return text if text.strip() else ""
        except Exception as e:
            print(f"[OCR] Tesseract error: {e}")
    
    return ""


def extract_location_description(img: Image.Image) -> tuple:
    """Extract location description from CAD drawing using OpenAI Vision (single image)."""
    if not OPENAI_AVAILABLE or VISION_LLM is None:
        return "", ""
    
    try:
        # Convert image to base64
        buffered = BytesIO()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(buffered, format='JPEG', quality=85)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        structured_llm = VISION_LLM.with_structured_output(DrawingLocationInfo)
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": LOCATION_EXTRACTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        )
        
        result = structured_llm.invoke([message])
        return (
            result.drawing_id.strip() if result.drawing_id else "",
            result.location_description.strip() if result.location_description else ""
        )
        
    except Exception as e:
        print(f"[OpenAI] Location extraction error: {type(e).__name__}: {e}")
        return "", ""  # Return empty on any error - don't block processing


class BatchLocationInfo(BaseModel):
    """Batch extraction of locations from multiple drawings."""
    locations: List[str] = Field(
        description="List of location descriptions for each drawing, in order. Use empty string if not found."
    )


def extract_locations_batch(images: List[Image.Image]) -> List[str]:
    """Extract location descriptions from MULTIPLE drawings in a SINGLE API call."""
    if not OPENAI_AVAILABLE or VISION_LLM is None:
        return [""] * len(images)
    
    if len(images) == 0:
        return []
    
    # Single image? Use regular function
    if len(images) == 1:
        return [extract_location_description(images[0])]
    
    try:
        # Convert all images to base64
        image_contents = []
        for i, img in enumerate(images):
            buffered = BytesIO()
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Reduce quality for faster upload
            img.save(buffered, format='JPEG', quality=60)
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            image_contents.append({
                "type": "text", 
                "text": f"[DRAWING {i + 1}]"
            })
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        
        # Create batch prompt - STRICT
        batch_prompt = f"""TASK: Extract the location description from EACH of the {len(images)} CAD drawings below.

{LOCATION_EXTRACTION_PROMPT}

IMPORTANT FOR BATCH PROCESSING:
- Return EXACTLY {len(images)} results, one per drawing
- For EACH drawing, ONLY return text you can ACTUALLY READ in that specific image
- If a drawing has no visible location text, return "" (empty string) for that position
- DO NOT make up or guess identifiers - return "" if not clearly visible
- It is OKAY to return mostly empty strings if locations are not visible"""

        # Build message with all images
        content = [{"type": "text", "text": batch_prompt}] + image_contents
        
        structured_llm = VISION_LLM.with_structured_output(BatchLocationInfo)
        message = HumanMessage(content=content)
        
        result = structured_llm.invoke([message])
        
        # Ensure we have the right number of results
        locations = result.locations if result.locations else []
        while len(locations) < len(images):
            locations.append("")
        
        return locations[:len(images)]
        
    except Exception as e:
        print(f"Batch location extraction error: {e}")
        # Fallback: return empty strings
        return [""] * len(images)


def cleanup_temp_files():
    """Clean up temporary files (PDFs and images) but keep CSV results"""
    import shutil
    from pathlib import Path
    
    cleanup_folders = [
        "uploads",
        "output_images",
        "predicted_images",
        "predicted_images_web",
        "predicted_images2",
    ]
    
    for folder in cleanup_folders:
        folder_path = Path(folder)
        if folder_path.exists():
            try:
                shutil.rmtree(folder_path)
                os.makedirs(folder_path, exist_ok=True)  # Recreate empty folder
            except Exception as e:
                print(f"Cleanup error for {folder}: {e}")


def get_sheet_number(page) -> str:
    """Extract sheet number - looks for sheet number near title block keywords"""
    try:
        x0, y0, x1, y1 = page.bbox
        w, h = x1 - x0, y1 - y0
        
        # Sheet number pattern: Letter(s) + digits, common formats
        # A001, A-001, A.001, A001-A, EL-001, etc.
        sheet_pattern = re.compile(r'\b([A-Z]{1,4}[-.]?\d{2,4}(?:[-.]?[A-Z0-9]{1,2})?)\b', re.IGNORECASE)
        
        # Keywords that typically appear near sheet numbers in title blocks
        sheet_keywords = ['SHEET', 'SHT', 'DWG', 'DRAWING', 'NO.', 'NO:', 'NUMBER', '#', 'PAGE']
        
        # Define regions to search (title blocks are usually in corners)
        # Format: (left_pct, top_pct, right_pct, bottom_pct, priority)
        search_regions = [
            (0.65, 0.85, 1.0, 1.0, 10),   # Bottom-right corner (highest priority - most common)
            (0.0, 0.85, 0.35, 1.0, 8),    # Bottom-left corner
            (0.65, 0.0, 1.0, 0.15, 6),    # Top-right corner
            (0.0, 0.0, 0.35, 0.15, 5),    # Top-left corner
        ]
        
        best_match = None
        best_score = 0
        
        for left_pct, top_pct, right_pct, bottom_pct, region_priority in search_regions:
            try:
                crop = page.crop((
                    x0 + w * left_pct,
                    y0 + h * top_pct,
                    x0 + w * right_pct,
                    y0 + h * bottom_pct
                ))
                text = crop.extract_text()
                
                if not text:
                    continue
                
                text_upper = text.upper()
                
                # Check if this region has sheet-related keywords
                has_keyword = any(kw in text_upper for kw in sheet_keywords)
                
                # Find all potential sheet numbers
                matches = sheet_pattern.findall(text)
                
                for match in matches:
                    sheet = match.strip().upper()
                    
                    # Skip if too short or too long
                    if len(sheet) < 2 or len(sheet) > 10:
                        continue
                    
                    # Calculate score for this match
                    score = region_priority
                    
                    # Bonus if near keywords
                    if has_keyword:
                        score += 20
                    
                    # Bonus for common sheet prefixes (A=Architectural, E=Electrical, etc.)
                    if sheet[0] in 'AESMPCGLD':
                        score += 10
                    
                    # Bonus if starts with letter and has 2-3 digits (most common format)
                    if re.match(r'^[A-Z]\d{2,3}$', sheet):
                        score += 15
                    
                    # Penalty for patterns that look like tags (e.g., PT-103 is likely a tag, not sheet)
                    if re.match(r'^[A-Z]{2,}_?[A-Z]*-?\d+$', sheet):
                        # This looks more like a material tag (PT-103, WC-12, etc.)
                        score -= 10
                    
                    if score > best_score:
                        best_score = score
                        best_match = sheet
                        
            except Exception:
                continue
        
        if best_match:
            return best_match
            
    except Exception as e:
        print(f"[SHEET] Error extracting sheet number: {e}")
    
    return "N/A"


async def process_pdf_with_job(job: JobState):
    """Process PDF as a background job - independent of WebSocket"""
    job.status = JobStatus.RUNNING
    print(f"[JOB-{job.job_id}] Starting PDF processing: {job.pdf_path}")
    print(f"[JOB-{job.job_id}] TAG_DESCRIPTIONS has {len(TAG_DESCRIPTIONS)} tags: {list(TAG_DESCRIPTIONS.keys())}")
    results = []
    target_tags = list(TAG_DESCRIPTIONS.keys())
    print(f"[JOB-{job.job_id}] Target tags for detection: {target_tags}")
    
    await job.broadcast({"type": "start", "message": "Starting extraction..."})
    
    def clean_tag_text(t):
        """Remove all non-alphanumeric characters for fuzzy matching."""
        import re
        return re.sub(r'[^A-Z0-9]', '', str(t).upper())
    
    # 1. Create a map of cleaned tags to original tags for inference
    # This allows matching "MC1" to "MC-1"
    CLEANED_MAPPING = {}
    for t in target_tags:
        ct = clean_tag_text(t)
        if ct:
            CLEANED_MAPPING[ct] = t
    
    CLEANED_TARGETS = list(CLEANED_MAPPING.keys())

    def process_ocr_and_tags(text, conf, sheet, page_num, location_desc):
        """Process OCR text to find tags (runs in thread)"""
        # Clean text for better matching (normalize spaces)
        text_clean = " ".join(text.split())
        
        # Split by whitespace and common delimiters to prevent greedy matching
        # This helps when OCR preserves spaces between tags
        tokens = re.split(r'[\s;,]+', text_clean)
        
        # Find tags in individual tokens first (prevents greedy cross-token matching)
        found_tags_raw = []
        for token in tokens:
            # Skip very short or very long tokens
            if len(token) < 2 or len(token) > 30:
                continue
            matches = TAG_PATTERN.findall(token)
            found_tags_raw.extend(matches)
        
        # Also try matching on the full text for tags that might span delimiters
        # But limit match length to prevent super-greedy matches
        full_matches = TAG_PATTERN.findall(text_clean)
        for match in full_matches:
            # Only add if it's a reasonable tag length (not a huge concatenation)
            if len(match) <= 20 and match not in found_tags_raw:
                found_tags_raw.append(match)
        
        found_tags = []
        # Normalize found tags (remove spaces)
        seen_raw_tags = set()
        for t in found_tags_raw:
            norm_tag = t.replace(" ", "")
            if norm_tag not in seen_raw_tags:
                found_tags.append(norm_tag)
                seen_raw_tags.add(norm_tag)

        matched_tags = []
        drawing_results = []
        
        # Deduplicate: Only report each unique original tag ONCE per drawing
        seen_tags_in_drawing = set()
        
        # Capture OCR text sample for debugging UI
        ocr_sample = text_clean[:50] + "..." if len(text_clean) > 50 else text_clean
        
        for tag in found_tags:
            # Clean the tag found by OCR for comparison
            # Example: OCR finds "MC1", we clean it to "MC1"
            tag_cleaned = clean_tag_text(tag)
            
            # Try variations for common OCR character confusion
            # O ↔ 0, 1 ↔ I ↔ l (lowercase L), A ↔ 4
            def generate_ocr_variations(s):
                """Generate all variations of a string with common OCR substitutions."""
                # Normalize to uppercase first (clean_tag_text already does this, but be safe)
                s = s.upper()
                variations = {s}
                
                # O ↔ 0
                if 'O' in s:
                    variations.add(s.replace('O', '0'))
                if '0' in s:
                    variations.add(s.replace('0', 'O'))
                
                # 1 ↔ I (OCR often confuses 1 and I and lowercase L)
                if '1' in s:
                    variations.add(s.replace('1', 'I'))
                if 'I' in s:
                    variations.add(s.replace('I', '1'))
                # L can also look like 1 in some fonts
                if 'L' in s:
                    variations.add(s.replace('L', '1'))
                
                # A ↔ 4 (OCR sometimes confuses A and 4)
                if 'A' in s:
                    variations.add(s.replace('A', '4'))
                if '4' in s:
                    variations.add(s.replace('4', 'A'))
                
                # Also try combinations (e.g., both I→1 and A→4)
                for v in list(variations):
                    if '1' in v:
                        variations.add(v.replace('1', 'I'))
                    if 'I' in v:
                        variations.add(v.replace('I', '1'))
                    if 'A' in v:
                        variations.add(v.replace('A', '4'))
                    if '4' in v:
                        variations.add(v.replace('4', 'A'))
                
                return variations
            
            variations = generate_ocr_variations(tag_cleaned)
            
            # Check against CLEANED_TARGETS
            match_found = False
            matched_original_tag = None
            
            # Step 1: Try exact match first
            for v in variations:
                if v in CLEANED_TARGETS:
                    matched_original_tag = CLEANED_MAPPING[v]
                    match_found = True
                    break
            
            # Step 2: If no exact match, try SUBSTRING matching
            # Handles cases like CA_WC-12 (OCR) matching A_WC-12 (target)
            # OCR might add extra characters due to shapes/borders
            if not match_found:
                for v in variations:
                    for target in CLEANED_TARGETS:
                        # Check if target is contained in OCR result (OCR added prefix)
                        # e.g., CAWC12 contains AWC12
                        if target in v and len(target) >= 3:
                            matched_original_tag = CLEANED_MAPPING[target]
                            match_found = True
                            break
                        # Check if OCR result is contained in target (OCR missed prefix)
                        # e.g., WC12 is in AWC12
                        if v in target and len(v) >= 3:
                            matched_original_tag = CLEANED_MAPPING[target]
                            match_found = True
                            break
                    if match_found:
                        break
            
            if match_found and matched_original_tag:
                # Deduplication check on original tag
                if matched_original_tag in seen_tags_in_drawing:
                    continue  # Already added, skip
                    
                seen_tags_in_drawing.add(matched_original_tag)
                matched_tags.append(matched_original_tag)
                
                # Use a special debug flag in the result to show in UI log
                drawing_results.append({
                    "material_type": TAG_DESCRIPTIONS.get(matched_original_tag, "Unknown"),
                    "tag": matched_original_tag,
                    "sheet": sheet,
                    "page": page_num,
                    "description": "", # Will be drawing_id from OpenAI
                    "location": location_desc,
                    "confidence": f"{conf:.0%}",
                    "_debug_match": True 
                })
            
            if not match_found and len(found_tags_raw) < 5:
                 # Return a "debug" result that isn't a real match but logs to UI
                 drawing_results.append({
                     "_debug_ignored": True,
                     "tag": tag.upper(),
                     "variations": variations,
                     "ocr_text": ocr_sample
                 })

        return matched_tags, drawing_results
    
    try:
        with pdfplumber.open(job.pdf_path) as pdf:
            total = min(job.end_page, len(pdf.pages)) - job.start_page + 1
            job.total_pages = total
            
            await job.broadcast({"type": "info", "total_pages": total})
            
            for idx, page_num in enumerate(range(job.start_page, min(job.end_page + 1, len(pdf.pages) + 1))):
                job.current_page = page_num
                
                # Job continues even with 0 subscribers (60s timeout handles cancellation)
                # This allows for reconnection without losing progress
                
                print(f"[JOB-{job.job_id}] Starting page {page_num} (subscribers: {len(job.subscribers)})")
                
                # Get page AND sheet number in thread (both can block)
                def get_page_and_sheet():
                    page = pdf.pages[page_num - 1]
                    sheet = get_sheet_number(page)
                    return page, sheet
                
                page, sheet = await asyncio.to_thread(get_page_and_sheet)
                print(f"[JOB-{job.job_id}] Page {page_num} loaded, sheet: {sheet}")
                
                # Send page start
                await job.broadcast({
                    "type": "page_start",
                    "page": page_num,
                    "sheet": sheet,
                    "progress": idx + 1,
                    "total": total
                })
                
                # Initialize page tracking for reconnection replay
                job.processed_pages[page_num] = {
                    'sheet': sheet,
                    'full_page_url': None,
                    'drawings': [],
                    'total_drawings': 0
                }
                job.current_drawing_index = 0
                
                # Detect drawings - run in thread to not block heartbeat
                await job.broadcast({"type": "log", "level": "info", "message": f"Detecting drawings on page {page_num}..."})
                print(f"[JOB-{job.job_id}] Starting detection for page {page_num}")
                try:
                    full_page_img, cropped_drawings, drawing_data = await asyncio.to_thread(
                        detect_drawings_on_page, page, page_num
                    )
                    print(f"[JOB-{job.job_id}] Detection complete: {len(cropped_drawings) if cropped_drawings else 0} drawings")
                except Exception as e:
                    print(f"[JOB-{job.job_id}] Detection ERROR: {type(e).__name__}: {e}")
                    full_page_img, cropped_drawings, drawing_data = None, [], []
                
                if full_page_img:
                    # Save full page image to disk and send URL (no size limit issues)
                    full_page_resized = full_page_img.copy()
                    if full_page_resized.width > 1400:
                        ratio = 1400 / full_page_resized.width
                        new_height = int(full_page_resized.height * ratio)
                        full_page_resized = full_page_resized.resize((1400, new_height), Image.Resampling.LANCZOS)
                    
                    # Save to disk and get URL
                    full_page_filename = f"{job.job_id}-page-{page_num}-full.jpg"
                    full_page_url = save_image_to_disk(full_page_resized, full_page_filename, quality=75)
                    print(f"[JOB-{job.job_id}] Saved full page to {full_page_url}")
                    
                    # Track for reconnection replay
                    job.processed_pages[page_num]['full_page_url'] = full_page_url
                    job.processed_pages[page_num]['total_drawings'] = len(cropped_drawings)
                    
                    # Send URL instead of base64 (tiny payload, no connection issues)
                    await job.broadcast({
                        "type": "full_page",
                        "image_url": full_page_url,  # URL instead of base64
                        "page": page_num,
                        "sheet": sheet,
                        "drawing_count": len(cropped_drawings)
                    })
                
                if cropped_drawings:
                    num_drawings = len(cropped_drawings)
                    
                    # Process each drawing ONE BY ONE - NO PARALLEL
                    for i, crop_data in enumerate(cropped_drawings):
                        # Track current drawing index for reconnection
                        job.current_drawing_index = i
                        
                        # Job continues even with 0 subscribers (allows reconnection)
                        cropped_img = crop_data["image"]
                        conf = crop_data["confidence"]
                        
                        # Save drawing image to disk and send URL (no size limit issues)
                        drawing_img = cropped_img.copy()
                        if drawing_img.width > 1200 or drawing_img.height > 1200:
                            drawing_img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                        
                        # Save to disk and get URL
                        drawing_filename = f"{job.job_id}-page-{page_num}-drawing-{i+1}.jpg"
                        drawing_url = save_image_to_disk(drawing_img, drawing_filename, quality=80)
                        
                        # Track for reconnection replay
                        job.processed_pages[page_num]['drawings'].append({
                            'index': i + 1,
                            'image_url': drawing_url,
                            'confidence': f"{conf:.0%}",
                            'bbox': crop_data["bbox"]
                        })
                        
                        # Send URL instead of base64 (tiny payload, no connection issues)
                        await job.broadcast({
                            "type": "drawing",
                            "index": i + 1,
                            "total_drawings": num_drawings,
                            "image_url": drawing_url,  # URL instead of base64
                            "confidence": f"{conf:.0%}",
                            "bbox": crop_data["bbox"]
                        })
                        
                        await job.broadcast({
                            "type": "log",
                            "level": "info",
                            "message": f"Analyzing drawing {i + 1}/{num_drawings}..."
                        })
                        
                        # Step 1: Run OCR First (sequential)
                        try:
                            await job.broadcast({"type": "log", "level": "info", "message": f"Running OCR..."})
                            print(f"[JOB-{job.job_id}] Starting OCR for drawing {i+1}")
                            text = await asyncio.to_thread(ocr_image, cropped_img)
                            print(f"[JOB-{job.job_id}] OCR complete: {len(text)} chars")
                            
                            # Log raw OCR for debugging
                            cleaned_snippet = text.replace('\n', ' ')[:100]
                            await job.broadcast({
                                "type": "log",
                                "level": "info", 
                                "message": f"RAW OCR OUT: {cleaned_snippet}..."
                            })

                        except Exception as e:
                            print(f"[JOB-{job.job_id}] OCR ERROR: {type(e).__name__}: {e}")
                            text = ""
                        
                        # Step 2: Check for tags
                        # Run OCR and tag matching in thread
                        location_desc = "" # Default empty
                        temp_results = []
                        
                        matched_tags, drawing_results = await asyncio.to_thread(
                            process_ocr_and_tags, text, conf, sheet, page_num, "" # Pass empty location first
                        )
                        
                        # Step 3: Run OpenAI ONLY if tags found
                        valid_tags_exist = any(not r.get("_debug_ignored") for r in drawing_results)
                        
                        if valid_tags_exist and OPENAI_AVAILABLE:
                            try:
                                await job.broadcast({"type": "log", "level": "info", "message": f"Tags found! Extracting location..."})
                                print(f"[JOB-{job.job_id}] Starting OpenAI call for drawing {i+1}")
                                drawing_id, location_desc = await asyncio.to_thread(extract_location_description, cropped_img)
                                print(f"[JOB-{job.job_id}] OpenAI call complete: ID={drawing_id}, LOC={location_desc[:30]}")
                            except Exception as e:
                                print(f"[JOB-{job.job_id}] OpenAI ERROR: {type(e).__name__}: {e}")
                                drawing_id, location_desc = "", ""
                                await job.broadcast({"type": "log", "level": "warning", "message": f"Location extraction failed"})
                        elif not valid_tags_exist:
                             drawing_id, location_desc = "", ""
                             await job.broadcast({"type": "log", "level": "info", "message": f"No tags, skipping location extraction"})

                        # Step 4: Update results with location and drawing index
                        for r in drawing_results:
                            r['location'] = location_desc
                            r['description'] = drawing_id # Use alphanumeric code as description
                            r['drawing_index'] = i + 1
                            
                        # Process results including debug logs
                        valid_results = []
                        for result in drawing_results:
                            if result.get("_debug_ignored"):
                                # This is a debug message for ignored tags
                                await job.broadcast({
                                    "type": "log", 
                                    "level": "warning", 
                                    "message": f"🔍 OCR saw '{result['tag']}' but matched none of {len(target_tags)} tags (OCR: '{result['ocr_text']}')"
                                })
                            elif result.get("_debug_match"):
                                # This is a successful match debug info
                                # Remove the internal flag before adding
                                del result["_debug_match"]
                                valid_results.append(result)
                                await job.broadcast({
                                    "type": "log", 
                                    "level": "success", 
                                    "message": f"✅ MATCHED: {result['tag']} in drawing {i+1} (Page {page_num})"
                                })
                                
                                # Send structured event for UI update
                                await job.broadcast({
                                    "type": "tag_match",
                                    "page": page_num,
                                    "drawing_index": i + 1,
                                    "tag": result['tag'],
                                    "description": result.get('description', '')
                                })
                            else:
                                valid_results.append(result)
                        
                        if valid_results:
                            job.results.extend(valid_results)
                            
                        if location_desc:
                            await job.broadcast({
                                "type": "log",
                                "level": "success",
                                "message": f"Drawing {i + 1} → {location_desc}"
                            })
                        
                        # Send OCR result with location
                        await job.broadcast({
                            "type": "ocr_result",
                            "drawing_index": i + 1,
                            "tags_found": matched_tags,
                            "text_preview": text[:100] if text else "No text found",
                            "location": location_desc
                        })
                        
                        # Add to results
                        results.extend(drawing_results)
                        job.results = results  # Update job state
                    
                    await job.broadcast({
                        "type": "log",
                        "level": "success",
                        "message": f"Page {page_num} complete - {num_drawings} drawings processed"
                    })
                else:
                    await job.broadcast({
                        "type": "log",
                        "level": "warning",
                        "message": f"No drawings detected on page {page_num}"
                    })
                
                # CRITICAL: Release memory after each page to prevent OOM
                del page
                if full_page_img:
                    del full_page_img
                if cropped_drawings:
                    del cropped_drawings
                if drawing_data:
                    del drawing_data
                
                # Force garbage collection to free memory immediately
                gc.collect()
                
                # Small delay to allow GC to complete
                await asyncio.sleep(0.1)
            
            # Filter valid results ONLY
            final_results = []
            seen_in_final = set() # Extra safety against duplicates
            
            for r in results:
                # 1. Skip debug objects
                if r.get("_debug_ignored") or r.get("_debug_match"):
                    continue
                
                # 2. Strict validation - must have tag and sheet
                if not r.get("tag") or not r.get("sheet"):
                    continue
                
                # 3. Final Deduplication
                # Create a signature: tag + sheet + page + drawing_index
                sig = f"{r['tag']}-{r['sheet']}-{r['page']}-{r.get('drawing_index', '0')}"
                if sig in seen_in_final:
                    continue
                seen_in_final.add(sig)
                final_results.append(r)
            
            # Save results
            csv_path = os.path.join(RESULTS_FOLDER, "results.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # New order: Material Type, Tag, Sheet, Page, Description, Location, Confidence
                writer.writerow(["Material Type", "Tag", "Sheet", "Page", "Description", "Location", "Confidence"])
                for r in final_results:
                    writer.writerow([
                        r.get("material_type", "Unknown"),
                        r.get("tag", ""),
                        r.get("sheet", ""),
                        r.get("page", ""),
                        r.get("description", ""),
                        r.get("location", ""),
                        r.get("confidence", "")
                    ])
            
            # Job completed successfully
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            
            await job.broadcast({
                "type": "complete",
                "total_tags": len(final_results),
                "results": final_results,
                "job_id": job.job_id
            })
            
            print(f"[JOB-{job.job_id}] COMPLETED - {len(final_results)} unique tags found")
            
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        print(f"[JOB-{job.job_id}] FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        await job.broadcast({"type": "error", "message": str(e)})
    
    # Auto-cleanup AFTER PDF is closed (outside the 'with' block)
    if AUTO_CLEANUP:
        await asyncio.sleep(0.5)  # Small delay to ensure file handles are released
        cleanup_temp_files()
        await job.broadcast({
            "type": "log",
            "level": "info",
            "message": "Auto-cleanup: Temporary files deleted (CSV results kept)"
        })
    
    # Cleanup job resources (images + PDF) after 5 minutes
    # Add extra delay to ensure PDF file handles are fully released
    await asyncio.sleep(300)
    # Additional delay to ensure PDF is fully closed
    await asyncio.sleep(5)
    cleanup_job_resources(job.job_id)  # Clean up saved images and PDF
    cleanup_job(job.job_id)


# === ROUTES ===

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    response = templates.TemplateResponse("index.html", {
        "request": request,
        "roboflow_available": ROBOFLOW_AVAILABLE,
        "openai_available": OPENAI_AVAILABLE,
        "tag_count": len(TAG_DESCRIPTIONS)
    })
    # Prevent browser caching to avoid connection issues on refresh
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.post("/upload-tags")
async def upload_tags(file: UploadFile = File(...)):
    """Upload CSV or XLSX file with tag list"""
    try:
        # Check file extension - support CSV, XLSX, and XLS
        filename = file.filename.lower()
        if not (filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.xls')):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "File must be CSV, XLSX, or XLS format"}
            )
        
        # Use pandas for robust CSV/XLSX reading
        import pandas as pd
        from io import BytesIO

        try:
            # Read file content once into memory
            content = await file.read()
            file_size = len(content)
            print(f"Received file: {file.filename}, Size: {file_size} bytes")
            
            if file_size < 10:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "File is too small or empty"}
                )

            # Use BytesIO to create a file-like object for pandas
            file_buffer = BytesIO(content)

            if filename.endswith('.csv'):
                try:
                    # Try reading with python engine to auto-detect separator
                    df = pd.read_csv(file_buffer, engine='python', sep=None, on_bad_lines='skip')
                except Exception as e:
                    print(f"CSV read error: {e}, trying default")
                    file_buffer.seek(0)
                    df = pd.read_csv(file_buffer)
                
                # If columns don't look like Tag/Description, assume headerless if 2 cols
                # Check for various casings
                cols_upper = [str(c).upper() for c in df.columns]
                
                if 'TAGS' not in cols_upper:
                     if len(df.columns) >= 2:
                         # Assume headerless
                         file_buffer.seek(0)
                         df = pd.read_csv(file_buffer, header=None, names=['Tags', 'Material Type'])
                     else:
                         # Try to force rename first column
                         df.rename(columns={df.columns[0]: 'Tags'}, inplace=True)
                else:
                    # Normalize headers
                    df.columns = [str(c).strip().title() for c in df.columns]
                    
            else: # .xlsx or .xls
                try:
                    # Try to read Excel file with openpyxl engine (for .xlsx)
                    if filename.endswith('.xlsx'):
                        df = pd.read_excel(file_buffer, engine='openpyxl')
                    else:
                        # For .xls files, try openpyxl first, then fallback
                        try:
                            df = pd.read_excel(file_buffer, engine='openpyxl')
                        except:
                            # Try without specifying engine (pandas will auto-detect)
                            file_buffer.seek(0)
                            df = pd.read_excel(file_buffer)
                except ImportError:
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "error": "Excel support not installed. Please install openpyxl: pip install openpyxl"}
                    )
                except Exception as e:
                    # Fallback: try reading without engine specification
                    try:
                        file_buffer.seek(0)
                        df = pd.read_excel(file_buffer)
                    except Exception as e2:
                        raise Exception(f"Failed to read Excel file: {str(e2)}. Please ensure the file is a valid Excel format (XLSX or XLS).")
                
                # Normalize column names (same logic as CSV)
                cols_upper = [str(c).upper() for c in df.columns]
                
                if 'TAGS' not in cols_upper:
                    if len(df.columns) >= 2:
                        # Assume headerless - read again without header
                        file_buffer.seek(0)
                        try:
                            if filename.endswith('.xlsx'):
                                df = pd.read_excel(file_buffer, engine='openpyxl', header=None, names=['Tags', 'Material Type'])
                            else:
                                df = pd.read_excel(file_buffer, header=None, names=['Tags', 'Material Type'])
                        except:
                            file_buffer.seek(0)
                            df = pd.read_excel(file_buffer, header=None, names=['Tags', 'Material Type'])
                    else:
                        # Try to force rename first column
                        df.rename(columns={df.columns[0]: 'Tags'}, inplace=True)
                else:
                    # Normalize headers (same as CSV)
                    df.columns = [str(c).strip().title() for c in df.columns]
                
            # Clean data
            df['Tags'] = df['Tags'].astype(str).str.strip().str.upper() # Ensure uppercase for tags
            if 'Material Type' in df.columns:
                df['Material Type'] = df['Material Type'].fillna('Unknown').astype(str).str.strip()
            else:
                df['Material Type'] = 'Unknown'
                
            global TAG_DESCRIPTIONS
            TAG_DESCRIPTIONS = pd.Series(df['Material Type'].values, index=df['Tags']).to_dict()
            
            # Remove any "nan" string keys or empty keys
            TAG_DESCRIPTIONS = {k: v for k, v in TAG_DESCRIPTIONS.items() if k and k.lower() != 'nan'}
            
            print(f"Loaded {len(TAG_DESCRIPTIONS)} tags: {list(TAG_DESCRIPTIONS.keys())[:5]}...")
            return {
                "success": True,
                "message": f"Loaded {len(TAG_DESCRIPTIONS)} tags successfully",
                "tags": list(TAG_DESCRIPTIONS.keys()),
                "count": len(TAG_DESCRIPTIONS)
            }
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error parsing file: {error_trace}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Error parsing file content: {str(e)}. Please ensure it's a valid CSV/XLSX with 'Tag' and 'Description' columns."}
            )
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error processing file: {error_trace}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error processing file: {str(e)}"}
        )


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF file"""
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    with pdfplumber.open(file_path) as pdf:
        page_count = len(pdf.pages)
    
    return {"filename": file.filename, "path": file_path, "pages": page_count}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint with job-based processing"""
    await websocket.accept()
    active_connections.append(websocket)
    print(f"[WS] Connection accepted")
    
    current_job = None
    heartbeat_task = None
    
    async def heartbeat():
        """Send heartbeat - stops completely after job completion to save resources"""
        count = 0
        errors = 0
        heartbeat_interval = 1  # Start with 1 second during active job
        dummy_data = "X" * 100  # Reduced from 1KB to 100 bytes for less I/O
        completion_wait_count = 0  # Track time since completion
        max_wait_after_completion = 60  # Stop heartbeat 60 seconds after completion
        
        while True:
            await asyncio.sleep(heartbeat_interval)
            count += 1
            
            # Check job status (lightweight check)
            job_active = False
            job_completed = False
            if current_job:
                job_status = current_job.status
                job_active = job_status == JobStatus.RUNNING
                job_completed = job_status in [JobStatus.COMPLETED, JobStatus.FAILED]
                
                if job_completed:
                    completion_wait_count += 1
                    heartbeat_interval = 5  # 5 seconds after completion (just to allow result download)
                    
                    # Stop heartbeat completely after waiting period (zero resource usage)
                    if completion_wait_count >= max_wait_after_completion:
                        # Job is done, client had time to download results - stop heartbeat completely
                        break
                else:
                    heartbeat_interval = 1  # 1 second during active job
                    completion_wait_count = 0  # Reset counter
            else:
                # No job yet - keep minimal heartbeat (1 second) for connection health
                heartbeat_interval = 1
                completion_wait_count = 0
            
            try:
                # Send heartbeat if:
                # 1. Job is active (running)
                # 2. Job completed but within wait period (allows result download)
                # 3. No job yet (keeps connection alive for new jobs)
                if job_active or (job_completed and completion_wait_count < max_wait_after_completion) or not current_job:
                    await websocket.send_json({
                        "type": "heartbeat", 
                        "n": count,
                        "keep_alive": dummy_data
                    })
                
                # ONLY log if job is actively running (every 30 seconds to reduce I/O)
                if job_active and count % 30 == 0:
                    await websocket.send_json({
                        "type": "log",
                        "level": "info",
                        "message": f"⚡ Connection alive ({count}s)"
                    })
                    # Removed print statement - reduces disk I/O
                
                errors = 0
            except Exception as e:
                errors += 1
                # Only log errors if job is active (reduces I/O)
                if job_active:
                    print(f"[WS-HEARTBEAT] FAILED #{errors} at {count}s: {type(e).__name__}")
                if errors >= 3:
                    break
    
    try:
        # Start heartbeat IMMEDIATELY
        heartbeat_task = asyncio.create_task(heartbeat())
        print(f"[WS] Heartbeat started")
        
        # Main message loop
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            # Only log if job is active (reduces I/O)
            if current_job and current_job.status == JobStatus.RUNNING:
                print(f"[WS] Received: {action}")
            
            if action == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif action == "process":
                # CREATE NEW JOB
                job = create_job(
                    data.get("pdf_path"),
                    data.get("start_page", 1),
                    data.get("end_page", 10)
                )
                current_job = job
                
                # Subscribe this WebSocket to the job FIRST
                job.add_subscriber(websocket)
                print(f"[WS] Subscribed WebSocket to job {job.job_id}")
                
                # Small delay to ensure WebSocket is ready
                await asyncio.sleep(0.1)
                
                # Send job ID to client
                await websocket.send_json({
                    "type": "job_created",
                    "job_id": job.job_id,
                    "message": f"Job {job.job_id} created"
                })
                print(f"[WS] Sent job_created message to client")
                
                # Start job in background (independent of WebSocket)
                # Job will now broadcast to the subscribed WebSocket
                task = asyncio.create_task(process_pdf_with_job(job))
                JOB_TASKS[job.job_id] = task
                print(f"[WS] Job {job.job_id} started in background with {len(job.subscribers)} subscriber(s)")
            
            elif action == "subscribe":
                # SUBSCRIBE TO EXISTING JOB (reconnect scenario)
                job_id = data.get("job_id")
                job = get_job(job_id)
                
                if job:
                    current_job = job
                    job.add_subscriber(websocket)
                    
                    # Cancel any pending cancellation (someone reconnected!)
                    if job_id in CANCELLATION_TASKS:
                        CANCELLATION_TASKS[job_id].cancel()
                        del CANCELLATION_TASKS[job_id]
                        print(f"[WS] ✅ Cancelled scheduled cancellation - job {job_id} resumed!")
                    
                    # Send summary (NO image replay)
                    await job.send_summary(websocket)
                    
                    print(f"[WS] Subscribed to existing job {job_id} - will continue live")
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Job {job_id} not found"
                    })
                    print(f"[WS] Job {job_id} not found")
                
    except WebSocketDisconnect as e:
        print(f"[WS] Client disconnected: code={e.code if hasattr(e, 'code') else 'unknown'}")
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
                # Cancel any existing cancellation task
                if current_job.job_id in CANCELLATION_TASKS:
                    CANCELLATION_TASKS[current_job.job_id].cancel()
                
                # Schedule new cancellation task (10 minute delay - plenty of time to return)
                task = asyncio.create_task(schedule_job_cancellation(current_job.job_id, 600))
                CANCELLATION_TASKS[current_job.job_id] = task
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
        gc.collect()
        
        print(f"[WS] Connection closed and cleaned up")


@app.get("/api/jobs")
async def list_active_jobs():
    """List all active jobs (for reconnection)"""
    jobs = []
    for job_id, job in ACTIVE_JOBS.items():
        jobs.append({
            "job_id": job.job_id,
            "status": job.status,
            "current_page": job.current_page,
            "total_pages": job.total_pages,
            "start_page": job.start_page,
            "end_page": job.end_page,
            "results_count": len(job.results),
            "subscribers": len(job.subscribers),
            "started_at": job.started_at.isoformat() if job.started_at else None
        })
    return {"jobs": jobs}

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job"""
    job = get_job(job_id)
    if not job:
        return {"error": f"Job {job_id} not found"}
    
    return {
        "job_id": job.job_id,
        "status": job.status,
        "current_page": job.current_page,
        "total_pages": job.total_pages,
        "start_page": job.start_page,
        "end_page": job.end_page,
        "results_count": len(job.results),
        "subscribers": len(job.subscribers),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "error": job.error
    }

@app.get("/download")
async def download_csv():
    """Download results CSV"""
    csv_path = os.path.join(RESULTS_FOLDER, "results.csv")
    if os.path.exists(csv_path):
        return FileResponse(csv_path, filename="material_tracker_results.csv", media_type="text/csv")
    return {"error": "No results available"}




@app.post("/cleanup")
async def cleanup_endpoint():
    """Manually trigger cleanup of temporary files"""
    try:
        cleanup_temp_files()
        return {"success": True, "message": "Cleanup completed. Temporary files deleted (CSV results kept)."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Scheduled cleanup task - runs daily at 9:17 PM
async def scheduled_cleanup_uploads():
    """Clean up uploads, static/images, and results folders daily at 9:17 PM"""
    total_deleted = 0
    
    # Clean up uploads folder
    uploads_path = Path(UPLOAD_FOLDER)
    if uploads_path.exists():
        try:
            deleted_count = 0
            for file_path in uploads_path.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        print(f"[SCHEDULED-CLEANUP] Deleted from uploads: {file_path.name}")
                    except Exception as e:
                        print(f"[SCHEDULED-CLEANUP] Error deleting {file_path.name}: {e}")
            
            total_deleted += deleted_count
            print(f"[SCHEDULED-CLEANUP] Completed cleanup of {UPLOAD_FOLDER} folder - {deleted_count} files deleted")
        except Exception as e:
            print(f"[SCHEDULED-CLEANUP] Error during uploads cleanup: {e}")
    
    # Clean up static/images folder
    images_path = Path(IMAGES_FOLDER)
    if images_path.exists():
        try:
            deleted_count = 0
            for file_path in images_path.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        print(f"[SCHEDULED-CLEANUP] Deleted from static/images: {file_path.name}")
                    except Exception as e:
                        print(f"[SCHEDULED-CLEANUP] Error deleting {file_path.name}: {e}")
            
            total_deleted += deleted_count
            print(f"[SCHEDULED-CLEANUP] Completed cleanup of {IMAGES_FOLDER} folder - {deleted_count} files deleted")
        except Exception as e:
            print(f"[SCHEDULED-CLEANUP] Error during static/images cleanup: {e}")
    
    # Clean up results folder
    results_path = Path(RESULTS_FOLDER)
    if results_path.exists():
        try:
            deleted_count = 0
            for file_path in results_path.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        print(f"[SCHEDULED-CLEANUP] Deleted from results: {file_path.name}")
                    except Exception as e:
                        print(f"[SCHEDULED-CLEANUP] Error deleting {file_path.name}: {e}")
            
            total_deleted += deleted_count
            print(f"[SCHEDULED-CLEANUP] Completed cleanup of {RESULTS_FOLDER} folder - {deleted_count} files deleted")
        except Exception as e:
            print(f"[SCHEDULED-CLEANUP] Error during results cleanup: {e}")
    
    print(f"[SCHEDULED-CLEANUP] Total files deleted: {total_deleted}")

def start_scheduled_cleanup():
    """Start the scheduled cleanup task"""
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        
        scheduler = AsyncIOScheduler()
        # Schedule cleanup to run daily at 9:17 PM
        scheduler.add_job(
            scheduled_cleanup_uploads,
            trigger=CronTrigger(hour=21, minute=17),  # 9:17 PM
            id='daily_cleanup_uploads',
            name='Daily cleanup of uploads, static/images, and results folders',
            replace_existing=True
        )
        scheduler.start()
        print("[SCHEDULER] Scheduled cleanup task started - will run daily at 9:17 PM (cleans uploads, static/images, and results)")
    except ImportError:
        print("[SCHEDULER] APScheduler not installed. Install with: pip install apscheduler")
        # Fallback: use asyncio-based simple scheduler
        asyncio.create_task(simple_scheduled_cleanup())

async def simple_scheduled_cleanup():
    """Simple asyncio-based scheduler (fallback if APScheduler not available)"""
    from datetime import datetime, timedelta
    
    while True:
        now = datetime.now()
        # Calculate next 9:17 PM
        target_time = now.replace(hour=21, minute=17, second=0, microsecond=0)
        if target_time <= now:
            target_time += timedelta(days=1)  # Next day
        
        wait_seconds = (target_time - now).total_seconds()
        print(f"[SCHEDULER] Next cleanup scheduled for {target_time.strftime('%Y-%m-%d %H:%M:%S')} (in {wait_seconds/3600:.1f} hours)")
        
        await asyncio.sleep(wait_seconds)
        await scheduled_cleanup_uploads()

# Start scheduler on app startup
@app.on_event("startup")
async def startup_event():
    """Start scheduled tasks on application startup"""
    start_scheduled_cleanup()

if __name__ == "__main__":
    import uvicorn
    # Start scheduler before running server
    start_scheduled_cleanup()
    # Disable WebSocket ping/pong to prevent disconnections when tab is inactive
    # Browser throttles inactive tabs, can't respond to pings, server would close connection
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        ws_ping_interval=None,  # Disable server pings
        ws_ping_timeout=None    # Disable ping timeout
    )
