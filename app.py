"""
CAD Material Tracker - FastAPI Web Application
Real-time UI with WebSockets for live updates
"""

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
            
            # If no subscribers left, mark job for cancellation
            if len(self.subscribers) == 0 and self.status == JobStatus.RUNNING:
                print(f"[JOB-{self.job_id}] ⚠️ No subscribers left - job will be cancelled")
                return True  # Signal to cancel
        return False
    
    async def broadcast(self, message: dict):
        """Broadcast message to all subscribers (no storage - we don't replay)"""
        # Broadcast to all connected clients
        dead_sockets = []
        for ws in self.subscribers[:]:  # Copy list to avoid modification during iteration
            try:
                await ws.send_json(message)
            except Exception as e:
                print(f"[JOB-{self.job_id}] Failed to send to subscriber: {e}")
                dead_sockets.append(ws)
        
        # Remove dead sockets
        for ws in dead_sockets:
            self.remove_subscriber(ws)
    
    async def send_summary(self, websocket: WebSocket):
        """Send job summary to reconnecting client - NO IMAGE REPLAY"""
        print(f"[JOB-{self.job_id}] Sending summary (no replay) to new subscriber")
        
        # Count what we've done
        pages_processed = self.current_page - self.start_page
        
        # Send summary
        try:
            await websocket.send_json({
                "type": "reconnect_summary",
                "job_id": self.job_id,
                "status": self.status,
                "current_page": self.current_page,
                "total_pages": self.total_pages,
                "pages_processed": pages_processed,
                "results_count": len(self.results),
                "message": f"Caught up to page {self.current_page}/{self.end_page}"
            })
            
            # Send current stats
            await websocket.send_json({
                "type": "log",
                "level": "success",
                "message": f"📊 Reconnected: Page {self.current_page}/{self.end_page} | {len(self.results)} tags found so far"
            })
            
            print(f"[JOB-{self.job_id}] Summary sent, will continue from page {self.current_page}")
        except Exception as e:
            print(f"[JOB-{self.job_id}] Failed to send summary: {e}")

# Global job registry
ACTIVE_JOBS: Dict[str, JobState] = {}
JOB_TASKS: Dict[str, asyncio.Task] = {}

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
    print(f"[JOB-{job_id}] Cleaned up")

# OpenAI Vision for location description extraction
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

class DrawingLocationInfo(BaseModel):
    """Extracted location information from a CAD drawing."""
    location_description: str = Field(
        description="The location identifier/description found in the drawing title block or header. Examples: 'E1 1/8 RCP - LEVEL 3 PART B', 'C3 KITCHEN E1 ELEVATION 2', 'B1 SECTION AT EAST OF GARAGE - NS'. Return empty string if not found."
    )

# Shared prompt for location extraction
LOCATION_EXTRACTION_PROMPT = """Extract the drawing title/location from this CAD drawing image.

Look carefully at these areas:
- Title block (usually bottom-right corner or top of page)
- Drawing header/label (large text near top)
- Sheet information area
- Any prominent text labels

Extract the main title or location identifier. Examples:
- "E1 ELEVATION WEST"
- "KITCHEN ELEVATION" 
- "LEVEL 3 PLAN"
- "A101 FLOOR PLAN"
- "SECTION AT GARAGE"
- "DOOR SCHEDULE"
- "TYPICAL MOUNTING HEIGHTS"

Return the text you see. If multiple text elements exist, combine them into a meaningful title.
Only return empty string if there is truly NO title or location text visible in the image."""

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
        print(f"OpenAI Vision enabled with gpt-4o model")
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
AUTO_CLEANUP = os.getenv("AUTO_CLEANUP", "false").lower() == "true"  # Set AUTO_CLEANUP=true in .env to enable

# Ensure folders exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, RESULTS_FOLDER, "static", "templates"]:
    os.makedirs(folder, exist_ok=True)

# Prevent PIL crash
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 500_000_000

# Tag patterns
TAG_PATTERN = re.compile(r'\b([A-Z]{1,3}[-_]?\d{1,2})\b', re.IGNORECASE)
SHEET_PATTERN = re.compile(r'\b([A-Z]\d{3})\b')

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

# Tesseract setup - works on both Windows and Linux
try:
    import pytesseract
    import shutil
    
    # Common installation paths for Windows and Linux
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",  # Windows default
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",  # Windows x86
        "/usr/local/bin/tesseract",  # Linux (compiled from source on EC2)
        "/usr/bin/tesseract",  # Linux (package manager)
        "/opt/homebrew/bin/tesseract",  # macOS Homebrew ARM
        "/usr/local/Cellar/tesseract/*/bin/tesseract",  # macOS Homebrew Intel
    ]
    
    tesseract_found = False
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            tesseract_found = True
            print(f"Tesseract found at: {path}")
            break
    
    # Fallback: check if tesseract is in system PATH
    if not tesseract_found:
        tesseract_in_path = shutil.which("tesseract")
        if tesseract_in_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_in_path
            print(f"Tesseract found in PATH: {tesseract_in_path}")
        else:
            print("WARNING: Tesseract not found! OCR will not work.")
except ImportError:
    print("WARNING: pytesseract not installed")

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
            for pred in result['predictions']:
                if pred.get('class') == 'drawing' and pred.get('confidence', 0) > 0.8:
                    x, y = pred['x'] * scale_x, pred['y'] * scale_y
                    w, h = pred['width'] * scale_x, pred['height'] * scale_y
                    x1, y1 = max(0, x - w/2), max(0, y - h/2)
                    x2, y2 = min(orig_width, x + w/2), min(orig_height, y + h/2)
                    
                    if x2 > x1 and y2 > y1:
                        drawings.append({
                            "bbox": (x1, y1, x2, y2),
                            "confidence": pred.get('confidence', 0)
                        })
                        # Crop the drawing region
                        cropped = img_original.crop((int(x1), int(y1), int(x2), int(y2)))
                        cropped_images.append({
                            "image": cropped,
                            "confidence": pred.get('confidence', 0),
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
    """OCR an image with advanced preprocessing for CAD drawings - optimized for material tags"""
    try:
        from pytesseract import image_to_string
        
        # Convert to grayscale
        if img.mode != 'L':
            img_gray = img.convert('L')
        else:
            img_gray = img.copy()
        
        # CRITICAL: Scale up small images (OCR works better on larger text)
        width, height = img_gray.size
        min_dimension = 1200  # Minimum dimension for good OCR
        if width < min_dimension or height < min_dimension:
            scale = max(min_dimension / width, min_dimension / height)
            new_size = (int(width * scale), int(height * scale))
            img_gray = img_gray.resize(new_size, Image.Resampling.LANCZOS)
        
        # Aggressive contrast enhancement
        img_enhanced = ImageEnhance.Contrast(img_gray).enhance(3.5)
        img_enhanced = ImageEnhance.Brightness(img_enhanced).enhance(1.1)
        
        # Auto-contrast for better text visibility
        img_enhanced = ImageOps.autocontrast(img_enhanced, cutoff=5)
        
        # Sharpen to make text edges crisp
        img_enhanced = img_enhanced.filter(ImageFilter.SHARPEN)
        img_enhanced = img_enhanced.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # Remove noise
        img_enhanced = img_enhanced.filter(ImageFilter.MedianFilter(size=3))
        
        # Try multiple OCR strategies and combine
        results = []
        char_whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_'
        
        # Strategy 1: Sparse text with character whitelist (best for material tags)
        try:
            text1 = image_to_string(
                img_enhanced,
                config=f'--psm 11 --oem 3 -c tessedit_char_whitelist={char_whitelist}'
            )
            if text1.strip():
                results.append(text1.strip())
        except Exception as e:
            print(f"[OCR] Strategy 1 failed: {e}")
        
        # Strategy 2: Single word mode (for individual tags like "A1", "B2")
        try:
            text2 = image_to_string(
                img_enhanced,
                config=f'--psm 8 --oem 3 -c tessedit_char_whitelist={char_whitelist}'
            )
            if text2.strip() and text2.strip() not in results:
                results.append(text2.strip())
        except:
            pass
        
        # Strategy 3: Use LSTM neural network (slower but more accurate)
        try:
            text3 = image_to_string(
                img_enhanced,
                config=f'--psm 11 --oem 1 -c tessedit_char_whitelist={char_whitelist}'
            )
            if text3.strip() and text3.strip() not in results:
                results.append(text3.strip())
        except:
            pass
        
        # Strategy 4: Try inverted image (some CAD drawings have white text on dark)
        try:
            img_inverted = ImageOps.invert(img_enhanced)
            text4 = image_to_string(
                img_inverted,
                config=f'--psm 11 --oem 3 -c tessedit_char_whitelist={char_whitelist}'
            )
            if text4.strip() and text4.strip() not in results:
                results.append(text4.strip())
        except:
            pass
        
        # Combine all unique results
        combined = ' '.join(set(results)).strip()
        
        # Clean up common OCR misreads
        replacements = {
            '|': 'I',  # Pipe to I
            '0': 'O',  # Zero to O (context-dependent, but helps)
            '1': 'I',  # One to I (for tags like "A1" vs "AI")
            '5': 'S',  # Five to S
            '8': 'B',  # Eight to B
        }
        for old, new in replacements.items():
            combined = combined.replace(old, new)
        
        return combined if combined else ""
        
    except Exception as e:
        print(f"[OCR] Error: {type(e).__name__}: {e}")
        # Fallback to simple OCR
        try:
            if img.mode != 'L':
                img = img.convert('L')
            img = ImageEnhance.Contrast(img).enhance(2.0)
            img = img.filter(ImageFilter.SHARPEN)
            from pytesseract import image_to_string
            return image_to_string(img, config='--psm 11 --oem 3')
        except:
            return ""


def extract_location_description(img: Image.Image) -> str:
    """Extract location description from CAD drawing using OpenAI Vision (single image)."""
    if not OPENAI_AVAILABLE or VISION_LLM is None:
        return ""
    
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
        return result.location_description.strip() if result.location_description else ""
        
    except Exception as e:
        print(f"[OpenAI] Location extraction error: {type(e).__name__}: {e}")
        return ""  # Return empty on any error - don't block processing


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
    """Extract sheet number"""
    try:
        x0, y0, x1, y1 = page.bbox
        w, h = x1 - x0, y1 - y0
        crop = page.crop((x0 + w * 0.6, y0 + h * 0.8, x1, y1))
        text = crop.extract_text()
        if text:
            match = SHEET_PATTERN.search(text)
            if match:
                return match.group(1)
    except:
        pass
    return "N/A"


async def process_pdf_with_job(job: JobState):
    """Process PDF as a background job - independent of WebSocket"""
    job.status = JobStatus.RUNNING
    print(f"[JOB-{job.job_id}] Starting PDF processing: {job.pdf_path}")
    results = []
    target_tags = list(TAG_DESCRIPTIONS.keys())
    
    await job.broadcast({"type": "start", "message": "Starting extraction..."})
    
    def process_ocr_and_tags(text, conf, sheet, page_num, location_desc):
        """Process OCR text to find tags (runs in thread)"""
        found_tags = TAG_PATTERN.findall(text)
        matched_tags = []
        drawing_results = []
        
        for tag in found_tags:
            tag_upper = tag.upper()
            for t in [tag_upper, tag_upper.replace('O', '0'), tag_upper.replace('0', 'O')]:
                if t in target_tags:
                    matched_tags.append(t)
                    drawing_results.append({
                        "material": t,
                        "description": TAG_DESCRIPTIONS.get(t, "Unknown"),
                        "sheet": sheet,
                        "page": page_num,
                        "confidence": f"{conf:.0%}",
                        "location": location_desc
                    })
                    break
        
        return matched_tags, drawing_results
    
    try:
        with pdfplumber.open(job.pdf_path) as pdf:
            total = min(job.end_page, len(pdf.pages)) - job.start_page + 1
            job.total_pages = total
            
            await job.broadcast({"type": "info", "total_pages": total})
            
            for idx, page_num in enumerate(range(job.start_page, min(job.end_page + 1, len(pdf.pages) + 1))):
                job.current_page = page_num
                
                # CHECK: If no subscribers, cancel job immediately
                if len(job.subscribers) == 0:
                    print(f"[JOB-{job.job_id}] ❌ CANCELLED - No subscribers (user refreshed/closed)")
                    job.status = JobStatus.FAILED
                    job.error = "Cancelled - no active clients"
                    cleanup_job(job.job_id)
                    return
                
                print(f"[JOB-{job.job_id}] Starting page {page_num}")
                
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
                    # Send full page preview immediately
                    await job.broadcast({
                        "type": "full_page",
                        "image": image_to_base64(full_page_img, quality=70),
                        "page": page_num,
                        "sheet": sheet,
                        "drawing_count": len(cropped_drawings)
                    })
                
                if cropped_drawings:
                    num_drawings = len(cropped_drawings)
                    
                    # Process each drawing ONE BY ONE - NO PARALLEL
                    for i, crop_data in enumerate(cropped_drawings):
                        # CHECK: Cancel if no subscribers
                        if len(job.subscribers) == 0:
                            print(f"[JOB-{job.job_id}] ❌ CANCELLED mid-page - No subscribers")
                            job.status = JobStatus.FAILED
                            job.error = "Cancelled - no active clients"
                            cleanup_job(job.job_id)
                            return
                        
                        cropped_img = crop_data["image"]
                        conf = crop_data["confidence"]
                        
                        # Send drawing preview
                        await job.broadcast({
                            "type": "drawing",
                            "index": i + 1,
                            "total_drawings": num_drawings,
                            "image": image_to_base64(cropped_img, quality=85),
                            "confidence": f"{conf:.0%}",
                            "bbox": crop_data["bbox"]
                        })
                        
                        await job.broadcast({
                            "type": "log",
                            "level": "info",
                            "message": f"Analyzing drawing {i + 1}/{num_drawings}..."
                        })
                        
                        # Step 1: Get location from OpenAI (sequential)
                        location_desc = ""
                        if OPENAI_AVAILABLE:
                            try:
                                await job.broadcast({"type": "log", "level": "info", "message": f"Extracting location..."})
                                print(f"[JOB-{job.job_id}] Starting OpenAI call for drawing {i+1}")
                                location_desc = await asyncio.to_thread(extract_location_description, cropped_img)
                                print(f"[JOB-{job.job_id}] OpenAI call complete: {location_desc[:50] if location_desc else 'empty'}")
                            except Exception as e:
                                print(f"[JOB-{job.job_id}] OpenAI ERROR: {type(e).__name__}: {e}")
                                await job.broadcast({"type": "log", "level": "warning", "message": f"Location extraction failed"})
                        
                        # Step 2: Run OCR (sequential)
                        try:
                            await job.broadcast({"type": "log", "level": "info", "message": f"Running OCR..."})
                            print(f"[JOB-{job.job_id}] Starting OCR for drawing {i+1}")
                            text = await asyncio.to_thread(ocr_image, cropped_img)
                            print(f"[JOB-{job.job_id}] OCR complete: {len(text)} chars")
                        except Exception as e:
                            print(f"[JOB-{job.job_id}] OCR ERROR: {type(e).__name__}: {e}")
                            text = ""
                        
                        # Process tags
                        matched_tags, drawing_results = process_ocr_and_tags(
                            text, conf, sheet, page_num, location_desc
                        )
                        
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
            
            # Save results
            csv_path = os.path.join(RESULTS_FOLDER, "results.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Count", "Material", "Description", "Sheet", "Page", "Confidence", "Location"])
                for i, r in enumerate(results, 1):
                    writer.writerow([i, r["material"], r["description"], r["sheet"], r["page"], r["confidence"], r.get("location", "")])
            
            # Job completed successfully
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            
            await job.broadcast({
                "type": "complete",
                "total_tags": len(results),
                "results": results,
                "job_id": job.job_id
            })
            
            print(f"[JOB-{job.job_id}] COMPLETED - {len(results)} tags found")
            
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
    
    # Cleanup job after 5 minutes
    await asyncio.sleep(300)
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
        """Send heartbeat every 1 second - bulletproof connection"""
        count = 0
        errors = 0
        dummy_data = "X" * 1000  # 1KB keepalive payload
        
        while True:
            await asyncio.sleep(1)
            count += 1
            try:
                await websocket.send_json({
                    "type": "heartbeat", 
                    "n": count,
                    "keep_alive": dummy_data
                })
                
                if count % 10 == 0:
                    await websocket.send_json({
                        "type": "log",
                        "level": "info",
                        "message": f"⚡ Connection alive ({count}s)"
                    })
                    print(f"[WS-HEARTBEAT] {count} seconds, connection alive")
                
                errors = 0
            except Exception as e:
                errors += 1
                print(f"[WS-HEARTBEAT] FAILED #{errors} at {count}s: {type(e).__name__}")
                if errors >= 3:
                    print(f"[WS-HEARTBEAT] Stopping after {errors} failures")
                    break
    
    try:
        # Start heartbeat IMMEDIATELY
        heartbeat_task = asyncio.create_task(heartbeat())
        print(f"[WS] Heartbeat started")
        
        # Main message loop
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
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
                
                # Subscribe this WebSocket to the job
                job.add_subscriber(websocket)
                
                # Send job ID to client
                await websocket.send_json({
                    "type": "job_created",
                    "job_id": job.job_id,
                    "message": f"Job {job.job_id} created"
                })
                
                print(f"[WS] Created job {job.job_id}")
                
                # Start job in background (independent of WebSocket)
                task = asyncio.create_task(process_pdf_with_job(job))
                JOB_TASKS[job.job_id] = task
                print(f"[WS] Job {job.job_id} started in background")
            
            elif action == "subscribe":
                # SUBSCRIBE TO EXISTING JOB (reconnect scenario)
                job_id = data.get("job_id")
                job = get_job(job_id)
                
                if job:
                    current_job = job
                    job.add_subscriber(websocket)
                    
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
            should_cancel = current_job.remove_subscriber(websocket)
            if should_cancel:
                print(f"[WS] ❌ Last subscriber left - job {current_job.job_id} will be cancelled")
            else:
                print(f"[WS] Unsubscribed from job {current_job.job_id} (other clients still watching)")
        
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
