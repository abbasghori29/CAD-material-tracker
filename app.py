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

# Tesseract setup
try:
    import pytesseract
    for path in [r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
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


async def broadcast(message: dict):
    """Send message to all connected WebSocket clients"""
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except:
            pass


def detect_drawings_on_page(page, page_num: int):
    """Detect CAD drawings using Roboflow and return full page + cropped drawings"""
    if not ROBOFLOW_AVAILABLE or CLIENT is None:
        return None, [], None
    
    try:
        # High resolution for full page preview
        img_original = page.to_image(resolution=150).original.convert("RGB")
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
        return image_to_string(Image.open(buffered), config='--psm 11 --oem 3')
    except:
        return ""


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
    results = []
    target_tags = list(TAG_DESCRIPTIONS.keys())
    
    await broadcast({"type": "start", "message": "Starting extraction..."})
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total = min(end_page, len(pdf.pages)) - start_page + 1
            
            await broadcast({"type": "info", "total_pages": total})
            
            for idx, page_num in enumerate(range(start_page, min(end_page + 1, len(pdf.pages) + 1))):
                page = pdf.pages[page_num - 1]
                sheet = get_sheet_number(page)
                
                # Send page start
                await broadcast({
                    "type": "page_start",
                    "page": page_num,
                    "sheet": sheet,
                    "progress": idx + 1,
                    "total": total
                })
                
                # Detect drawings
                full_page_img, cropped_drawings, drawing_data = detect_drawings_on_page(page, page_num)
                
                if full_page_img:
                    # Send full page preview immediately
                    await broadcast({
                        "type": "full_page",
                        "image": image_to_base64(full_page_img, quality=70),
                        "page": page_num,
                        "sheet": sheet,
                        "drawing_count": len(cropped_drawings)
                    })
                
                # Send each cropped drawing one by one
                for i, crop_data in enumerate(cropped_drawings):
                    cropped_img = crop_data["image"]
                    conf = crop_data["confidence"]
                    
                    # Send cropped drawing preview
                    await broadcast({
                        "type": "drawing",
                        "index": i + 1,
                        "total_drawings": len(cropped_drawings),
                        "image": image_to_base64(cropped_img, quality=75),
                        "confidence": f"{conf:.0%}",
                        "bbox": crop_data["bbox"]
                    })
                    
                    # OCR the drawing
                    text = ocr_image(cropped_img)
                    found_tags = TAG_PATTERN.findall(text)
                    
                    matched_tags = []
                    for tag in found_tags:
                        tag_upper = tag.upper()
                        for t in [tag_upper, tag_upper.replace('O', '0'), tag_upper.replace('0', 'O')]:
                            if t in target_tags:
                                matched_tags.append(t)
                                results.append({
                                    "material": t,
                                    "description": TAG_DESCRIPTIONS.get(t, "Unknown"),
                                    "sheet": sheet,
                                    "page": page_num,
                                    "confidence": f"{conf:.0%}"
                                })
                                break
                    
                    # Send OCR result
                    await broadcast({
                        "type": "ocr_result",
                        "drawing_index": i + 1,
                        "tags_found": matched_tags,
                        "text_preview": text[:100] if text else "No text found"
                    })
                    
                    await asyncio.sleep(0.05)  # Small delay for smooth UI
                
                if not cropped_drawings:
                    await broadcast({
                        "type": "log",
                        "level": "warning",
                        "message": f"No drawings detected on page {page_num}"
                    })
                
                await asyncio.sleep(0.1)
            
            # Save results
            csv_path = os.path.join(RESULTS_FOLDER, "results.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Count", "Material", "Description", "Sheet", "Page", "Confidence"])
                for i, r in enumerate(results, 1):
                    writer.writerow([i, r["material"], r["description"], r["sheet"], r["page"], r["confidence"]])
            
            await broadcast({
                "type": "complete",
                "total_tags": len(results),
                "results": results
            })
            
    except Exception as e:
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
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("action") == "process":
                pdf_path = data.get("pdf_path")
                start_page = data.get("start_page", 1)
                end_page = data.get("end_page", 10)
                
                await process_pdf_realtime(pdf_path, start_page, end_page)
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


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
