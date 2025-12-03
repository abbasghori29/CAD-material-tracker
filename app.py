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
from typing import List
import base64
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
LOCATION_EXTRACTION_PROMPT = """Read the drawing title or location label from this CAD drawing image.

Look in the title block, header, or label area.

Return exactly what you see written - could be:
- With prefix: "E1 ELEVATION WEST", "A101 FLOOR PLAN", "S2 FRAMING"
- Without prefix: "KITCHEN ELEVATION", "LEVEL 3 PLAN", "SECTION AT GARAGE"

Just read what's there. Don't make up text that isn't visible.
If no title/location is visible, return empty string."""

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
    """Send message to all connected WebSocket clients - robust version"""
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
            # Force flush
            await asyncio.sleep(0)
        except Exception as e:
            print(f"[BROADCAST] Failed to send to client: {type(e).__name__}")
            disconnected.append(connection)
    
    # Clean up disconnected clients
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)
            print(f"[BROADCAST] Removed disconnected client")


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
    """OCR an image"""
    try:
        # Enhance
        if img.mode != 'L':
            img = img.convert('L')
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = img.filter(ImageFilter.SHARPEN).convert('RGB')
        
        buffered = BytesIO()
        img.save(buffered, format='PNG')
        buffered.seek(0)
        
        from pytesseract import image_to_string
        # PSM 11 = Sparse text - finds scattered text in any order (best for CAD drawings)
        return image_to_string(Image.open(buffered), config='--psm 11 --oem 3')
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


async def process_pdf_realtime(pdf_path: str, start_page: int, end_page: int):
    """Process PDF with real-time WebSocket updates"""
    print(f"[PROCESS] Starting PDF processing: {pdf_path}")
    results = []
    target_tags = list(TAG_DESCRIPTIONS.keys())
    
    await broadcast({"type": "start", "message": "Starting extraction..."})
    
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
        with pdfplumber.open(pdf_path) as pdf:
            total = min(end_page, len(pdf.pages)) - start_page + 1
            
            await broadcast({"type": "info", "total_pages": total})
            
            for idx, page_num in enumerate(range(start_page, min(end_page + 1, len(pdf.pages) + 1))):
                page = pdf.pages[page_num - 1]
                sheet = await asyncio.to_thread(get_sheet_number, page)
                
                # Send page start
                await broadcast({
                    "type": "page_start",
                    "page": page_num,
                    "sheet": sheet,
                    "progress": idx + 1,
                    "total": total
                })
                
                # Detect drawings - run in thread to not block heartbeat
                await broadcast({"type": "log", "level": "info", "message": f"Detecting drawings on page {page_num}..."})
                print(f"[PROCESS] Starting detection for page {page_num}")
                try:
                    full_page_img, cropped_drawings, drawing_data = await asyncio.to_thread(
                        detect_drawings_on_page, page, page_num
                    )
                    print(f"[PROCESS] Detection complete: {len(cropped_drawings) if cropped_drawings else 0} drawings")
                except Exception as e:
                    print(f"[PROCESS] Detection ERROR: {type(e).__name__}: {e}")
                    full_page_img, cropped_drawings, drawing_data = None, [], []
                
                if full_page_img:
                    # Send full page preview immediately
                    await broadcast({
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
                        cropped_img = crop_data["image"]
                        conf = crop_data["confidence"]
                        
                        # Send drawing preview
                        await broadcast({
                            "type": "drawing",
                            "index": i + 1,
                            "total_drawings": num_drawings,
                            "image": image_to_base64(cropped_img, quality=85),
                            "confidence": f"{conf:.0%}",
                            "bbox": crop_data["bbox"]
                        })
                        
                        await broadcast({
                            "type": "log",
                            "level": "info",
                            "message": f"Analyzing drawing {i + 1}/{num_drawings}..."
                        })
                        
                        # Step 1: Get location from OpenAI (sequential)
                        location_desc = ""
                        if OPENAI_AVAILABLE:
                            try:
                                await broadcast({"type": "log", "level": "info", "message": f"Extracting location..."})
                                print(f"[PROCESS] Starting OpenAI call for drawing {i+1}")
                                location_desc = await asyncio.to_thread(extract_location_description, cropped_img)
                                print(f"[PROCESS] OpenAI call complete: {location_desc[:50] if location_desc else 'empty'}")
                            except Exception as e:
                                print(f"[PROCESS] OpenAI ERROR: {type(e).__name__}: {e}")
                                await broadcast({"type": "log", "level": "warning", "message": f"Location extraction failed"})
                        
                        # Step 2: Run OCR (sequential)
                        try:
                            await broadcast({"type": "log", "level": "info", "message": f"Running OCR..."})
                            print(f"[PROCESS] Starting OCR for drawing {i+1}")
                            text = await asyncio.to_thread(ocr_image, cropped_img)
                            print(f"[PROCESS] OCR complete: {len(text)} chars")
                        except Exception as e:
                            print(f"[PROCESS] OCR ERROR: {type(e).__name__}: {e}")
                            text = ""
                        
                        # Process tags
                        matched_tags, drawing_results = process_ocr_and_tags(
                            text, conf, sheet, page_num, location_desc
                        )
                        
                        if location_desc:
                            await broadcast({
                                "type": "log",
                                "level": "success",
                                "message": f"Drawing {i + 1} → {location_desc}"
                            })
                        
                        # Send OCR result with location
                        await broadcast({
                            "type": "ocr_result",
                            "drawing_index": i + 1,
                            "tags_found": matched_tags,
                            "text_preview": text[:100] if text else "No text found",
                            "location": location_desc
                        })
                        
                        # Add to results
                        results.extend(drawing_results)
                    
                    await broadcast({
                        "type": "log",
                        "level": "success",
                        "message": f"Page {page_num} complete - {num_drawings} drawings processed"
                    })
                else:
                    await broadcast({
                        "type": "log",
                        "level": "warning",
                        "message": f"No drawings detected on page {page_num}"
                    })
            
            # Save results
            csv_path = os.path.join(RESULTS_FOLDER, "results.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Count", "Material", "Description", "Sheet", "Page", "Confidence", "Location"])
                for i, r in enumerate(results, 1):
                    writer.writerow([i, r["material"], r["description"], r["sheet"], r["page"], r["confidence"], r.get("location", "")])
            
            await broadcast({
                "type": "complete",
                "total_tags": len(results),
                "results": results
            })
            
    except Exception as e:
        print(f"[PROCESS] FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        await broadcast({"type": "error", "message": str(e)})
    
    # Auto-cleanup AFTER PDF is closed (outside the 'with' block)
    if AUTO_CLEANUP:
        await asyncio.sleep(0.5)  # Small delay to ensure file handles are released
        cleanup_temp_files()
        await broadcast({
            "type": "log",
            "level": "info",
            "message": "Auto-cleanup: Temporary files deleted (CSV results kept)"
        })


# === ROUTES ===

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "roboflow_available": ROBOFLOW_AVAILABLE,
        "openai_available": OPENAI_AVAILABLE,
        "tag_count": len(TAG_DESCRIPTIONS)
    })


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
    """WebSocket endpoint - NEVER times out, NEVER closes"""
    await websocket.accept()
    active_connections.append(websocket)
    print(f"[WS] Connection accepted")
    
    # Background tasks
    heartbeat_task = None
    processing_task = None
    
    async def heartbeat():
        """Send heartbeat every 1 second - keeps connection alive"""
        count = 0
        while True:
            await asyncio.sleep(1)
            count += 1
            try:
                await websocket.send_json({"type": "heartbeat", "n": count})
                if count % 10 == 0:  # Log every 10 seconds
                    print(f"[WS-HEARTBEAT] Sent {count} heartbeats, connection alive")
            except Exception as e:
                print(f"[WS-HEARTBEAT] FAILED at count {count}: {type(e).__name__}: {e}")
                break  # Connection is dead, stop heartbeat
    
    try:
        # Start heartbeat IMMEDIATELY
        heartbeat_task = asyncio.create_task(heartbeat())
        print(f"[WS] Heartbeat task created and started")
        
        # NO TIMEOUT - wait forever for messages
        while True:
            data = await websocket.receive_json()  # No timeout, wait forever
            print(f"[WS] Received message: {data.get('action', 'unknown')}")
            
            if data.get("action") == "ping":
                await websocket.send_json({"type": "pong"})
                print(f"[WS] Sent pong response")
            
            elif data.get("action") == "process":
                pdf_path = data.get("pdf_path")
                start_page = data.get("start_page", 1)
                end_page = data.get("end_page", 10)
                print(f"[WS] Starting processing task for {pdf_path}, pages {start_page}-{end_page}")
                
                # Run processing as background task
                processing_task = asyncio.create_task(
                    process_pdf_realtime(pdf_path, start_page, end_page)
                )
                print(f"[WS] Processing task created and running in background")
                
    except WebSocketDisconnect as e:
        print(f"[WS] Client disconnected: code={e.code if hasattr(e, 'code') else 'unknown'}")
    except Exception as e:
        print(f"[WS] UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        if heartbeat_task:
            heartbeat_task.cancel()
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
