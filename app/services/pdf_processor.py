"""
PDF Processor - Main PDF processing logic for extracting tags from CAD drawings.
"""

import os
import re
import csv
import gc
import asyncio
import time
import tempfile
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageOps

from app.core import config as app_config
from app.core.config import RESULTS_FOLDER, AUTO_CLEANUP, ALWAYS_PYMUPDF
from app.core.constants import TAG_PATTERN
from app.models.job import JobState, JobStatus
from app.services.image_utils import save_image_to_disk
from app.services.text_extractor import (
    image_coords_to_pdf_coords, 
    extract_text_from_pdf_region,
    get_sheet_number,
    is_pymupdf_available,
    get_fitz
)
from app.services.ai_service import (
    extract_location_description,
    extract_sheet_name_openai,
    get_roboflow_client,
    is_roboflow_available,
    is_openai_available
)
from app.services.ocr_service import ocr_image
from app.services.cleanup_service import cleanup_temp_files, cleanup_job_resources


def render_page_to_image(page, page_num: int, use_pymupdf: bool = True) -> Tuple[Image.Image, int]:
    """
    Render PDF page to image - returns image and resolution used
    
    Args:
        page: Either a PyMuPDF page (fitz.Page) or pdfplumber page
        page_num: Page number (1-indexed)
        use_pymupdf: If True, expects PyMuPDF page (faster), else pdfplumber page
    
    Returns:
        (img_original, resolution) tuple
    """
    render_start = time.time()
    fitz = get_fitz()
    
    if use_pymupdf and is_pymupdf_available():
        # Use PyMuPDF for faster rendering
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        max_dimension = max(page_width, page_height)
        
        # Adaptive resolution
        if max_dimension > 1728:  # ~24 inches at 72 dpi
            resolution = 100
            zoom = resolution / 72.0
        else:
            resolution = 150
            zoom = resolution / 72.0
        
        # Render page to pixmap (PyMuPDF is much faster)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_original = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    else:
        # Fallback to pdfplumber
        page_width, page_height = page.width, page.height
        max_dimension = max(page_width, page_height)
        
        if max_dimension > 1728:
            resolution = 100
        else:
            resolution = 150
        
        img_original = page.to_image(resolution=resolution).original.convert("RGB")
    
    img_original = ImageOps.exif_transpose(img_original)
    render_time = time.time() - render_start
    print(f"[RENDER] Page {page_num}: Rendered to image in {render_time:.2f}s (resolution: {resolution} DPI, PyMuPDF: {use_pymupdf and is_pymupdf_available()})")
    
    return img_original, resolution


def detect_drawings_on_image(img_original: Image.Image, page_num: int, resolution: int) -> Tuple[Optional[Image.Image], List[Dict], List[Dict]]:
    """
    Detect CAD drawings on already-rendered image using Roboflow and return annotated image + cropped drawings
    
    Args:
        img_original: PIL Image of the page (already rendered)
        page_num: Page number (1-indexed)
        resolution: Resolution used for rendering (for coordinate scaling)
        
    Returns:
        (img_annotated, cropped_images, drawings) tuple
    """
    if not is_roboflow_available() or get_roboflow_client() is None:
        return None, [], []
    
    try:
        start_time = time.time()
        CLIENT = get_roboflow_client()
        from app.core.config import ROBOFLOW_MODEL_ID
        
        orig_width, orig_height = img_original.size
        
        # Prepare for inference
        prep_start = time.time()
        img_for_inference = img_original.copy().convert("L").convert("RGB")
        img_resized = img_for_inference.resize((640, 640), Image.Resampling.LANCZOS)
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img_resized.save(temp_file.name, format='JPEG', quality=85)
        temp_file.close()
        prep_time = time.time() - prep_start
        print(f"[DETECT] Page {page_num}: Prepared image for inference in {prep_time:.2f}s")
        
        infer_start = time.time()
        result = CLIENT.infer(temp_file.name, model_id=ROBOFLOW_MODEL_ID)
        infer_time = time.time() - infer_start
        print(f"[DETECT] Page {page_num}: Roboflow inference took {infer_time:.2f}s")
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
            raw_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            final_detections = []
            while raw_detections:
                current = raw_detections.pop(0)
                final_detections.append(current)
                
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
                
                # Add padding to all four sides (5% of bounding box dimensions)
                padding_pct = 0.05
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                padding_x = bbox_width * padding_pct
                padding_y = bbox_height * padding_pct
                
                # Apply padding, ensuring we don't exceed image boundaries
                padded_x1 = max(0, int(x1 - padding_x))
                padded_y1 = max(0, int(y1 - padding_y))
                padded_x2 = min(orig_width, int(x2 + padding_x))
                padded_y2 = min(orig_height, int(y2 + padding_y))
                
                # Crop with padding
                cropped = img_original.crop((padded_x1, padded_y1, padded_x2, padded_y2))
                cropped_images.append({
                    "image": cropped,
                    "confidence": d['confidence'],
                    "bbox": (padded_x1, padded_y1, padded_x2, padded_y2)
                })
        
        # Draw boxes on full page image (RED color)
        post_start = time.time()
        img_annotated = img_original.copy()
        draw = ImageDraw.Draw(img_annotated)
        for i, d in enumerate(drawings):
            x1, y1, x2, y2 = d["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="#FF3B3B", width=4)
            # Draw number label
            draw.rectangle([x1, y1, x1+30, y1+25], fill="#FF3B3B")
            draw.text((x1+8, y1+3), str(i+1), fill="#FFFFFF")
        post_time = time.time() - post_start
        
        total_time = time.time() - start_time
        print(f"[DETECT] Page {page_num}: Post-processing took {post_time:.2f}s")
        print(f"[DETECT] Page {page_num}: TOTAL detection time: {total_time:.2f}s")
        
        return img_annotated, cropped_images, drawings
        
    except Exception as e:
        print(f"Detection error: {e}")
        return None, [], []


def clean_tag_text(t: str) -> str:
    """Remove all non-alphanumeric characters for fuzzy matching."""
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())


def generate_ocr_variations(s: str) -> set:
    """Generate all variations of a string with common OCR substitutions."""
    s = s.upper()
    variations = {s}
    
    # O ↔ 0
    if 'O' in s:
        variations.add(s.replace('O', '0'))
    if '0' in s:
        variations.add(s.replace('0', 'O'))
    
    # 1 ↔ I
    if '1' in s:
        variations.add(s.replace('1', 'I'))
    if 'I' in s:
        variations.add(s.replace('I', '1'))
    if 'L' in s:
        variations.add(s.replace('L', '1'))
    
    # A ↔ 4
    if 'A' in s:
        variations.add(s.replace('A', '4'))
    if '4' in s:
        variations.add(s.replace('4', 'A'))
    
    # Combinations
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


def process_ocr_and_tags(
    text: str, 
    conf: float, 
    sheet: str, 
    page_num: int, 
    location_desc: str, 
    is_ocr_text: bool = False,
    target_tags: List[str] = None,
    tag_descriptions: Dict[str, str] = None
) -> Tuple[List[str], List[Dict]]:
    """
    Process text to find tags.
    
    Args:
        text: Extracted text
        conf: Confidence score
        sheet: Sheet number
        page_num: Page number
        location_desc: Location description
        is_ocr_text: True if text came from OCR, False if from PyMuPDF
        target_tags: List of tags to search for
        tag_descriptions: Dictionary mapping tags to descriptions
    
    Returns:
        Tuple of (matched_tags, drawing_results)
    """
    if target_tags is None:
        target_tags = list(app_config.TAG_DESCRIPTIONS.keys())
    if tag_descriptions is None:
        tag_descriptions = app_config.TAG_DESCRIPTIONS
    
    # Create cleaned mapping
    CLEANED_MAPPING = {}
    for t in target_tags:
        ct = clean_tag_text(t)
        if ct:
            CLEANED_MAPPING[ct] = t
    
    CLEANED_TARGETS = list(CLEANED_MAPPING.keys())
    
    # Clean text for better matching
    text_clean = " ".join(text.split())
    
    # Find all tags using pattern
    found_tags_raw = []
    full_matches = TAG_PATTERN.findall(text_clean)
    for match in full_matches:
        if len(match) <= 20:
            found_tags_raw.append(match)
    
    found_tags = []
    for t in found_tags_raw:
        norm_tag = t.replace(" ", "")
        found_tags.append(norm_tag)

    matched_tags = []
    drawing_results = []
    
    # Count occurrences
    tag_counts = {}
    for tag in found_tags:
        normalized = tag.replace(" ", "").upper()
        tag_counts[normalized] = tag_counts.get(normalized, 0) + 1
    
    # Get unique tags
    unique_tags_to_process = {}
    for tag in found_tags:
        normalized = tag.replace(" ", "").upper()
        if normalized not in unique_tags_to_process:
            unique_tags_to_process[normalized] = tag
    
    # OCR text sample for debugging
    ocr_sample = text_clean[:50] + "..." if len(text_clean) > 50 else text_clean
    
    # Process each unique tag
    for normalized_tag, original_tag in unique_tags_to_process.items():
        occurrence_count = tag_counts.get(normalized_tag, 1)
        tag_cleaned = clean_tag_text(original_tag)
        
        # Generate variations based on text source
        if is_ocr_text:
            variations = generate_ocr_variations(tag_cleaned)
        else:
            variations = {tag_cleaned}
        
        # Check against targets
        match_found = False
        matched_original_tag = None
        
        if not is_ocr_text:
            # PyMuPDF: Try exact matching first
            if original_tag in target_tags:
                matched_original_tag = original_tag
                match_found = True
            elif not match_found:
                normalized = original_tag.replace(" ", "").upper()
                if normalized in target_tags:
                    matched_original_tag = normalized
                    match_found = True
        else:
            # OCR: Use cleaned matching
            for v in variations:
                if v in CLEANED_TARGETS:
                    matched_original_tag = CLEANED_MAPPING[v]
                    match_found = True
                    break

            # Substring matching for OCR
            if not match_found:
                for v in variations:
                    for target in CLEANED_TARGETS:
                        if target in v and len(target) >= 3:
                            matched_original_tag = CLEANED_MAPPING[target]
                            match_found = True
                            break
                        if v in target and len(v) >= 3:
                            matched_original_tag = CLEANED_MAPPING[target]
                            match_found = True
                            break
                    if match_found:
                        break
        
        if match_found and matched_original_tag:
            for occurrence_num in range(1, occurrence_count + 1):
                matched_tags.append(matched_original_tag)
                
                drawing_results.append({
                    "material_type": tag_descriptions.get(matched_original_tag, "Unknown"),
                    "tag": matched_original_tag,
                    "sheet": sheet,
                    "page": page_num,
                    "description": "",
                    "location": location_desc,
                    "confidence": f"{conf:.0%}",
                    "occurrence": occurrence_num,
                    "_debug_match": True 
                })
        
        if not match_found and len(found_tags_raw) < 5:
            drawing_results.append({
                "_debug_ignored": True,
                "tag": original_tag.upper(),
                "variations": variations,
                "ocr_text": ocr_sample
            })

    return matched_tags, drawing_results


# Position-based deduplication helpers
def find_all_tag_positions_cleaned(tag_text: str, word_positions: List) -> List[Tuple[float, float]]:
    """Find ALL positions of a tag in the word_positions list using CLEANED matching.
    
    Uses 'startswith' instead of exact match so that tags like A_WB-3
    are found even when PyMuPDF extracts them as A_WB-3/A (longer token).
    """
    if not word_positions:
        return []
    
    positions = []
    tag_cleaned = clean_tag_text(tag_text)
    
    for word, x, y in word_positions:
        word_cleaned = clean_tag_text(word)
        # startswith handles suffix cases like A_WB-3/A -> AWB3A
        if word_cleaned.startswith(tag_cleaned):
            # Check suffix to avoid partial number matches (e.g. matching 3 in 30)
            suffix = word_cleaned[len(tag_cleaned):]
            if suffix and suffix[0].isdigit():
                continue
            positions.append((x, y))
    
    return positions


def positions_are_same(pos1, pos2, tolerance=5) -> bool:
    """Check if two positions are the same tag (within tolerance)."""
    if pos1 is None or pos2 is None:
        return False
    distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    return distance < tolerance


def is_point_inside_bbox(point, bbox, tolerance=5) -> bool:
    """Check if a point is inside a bounding box (with small tolerance)."""
    if point is None or bbox is None:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    
    # Normalize bbox
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    return (x1 - tolerance <= x <= x2 + tolerance and 
            y1 - tolerance <= y <= y2 + tolerance)


def distance_to_region_center(tag_pos, region_bbox) -> float:
    """Calculate distance from tag position to region center."""
    if tag_pos is None or region_bbox is None:
        return float('inf')
    tag_x, tag_y = tag_pos
    x1, y1, x2, y2 = region_bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return ((tag_x - center_x) ** 2 + (tag_y - center_y) ** 2) ** 0.5


async def process_pdf_with_job(job: JobState):
    """Process PDF as a background job - independent of WebSocket"""
    job.status = JobStatus.RUNNING
    print(f"[JOB-{job.job_id}] Starting PDF processing: {job.pdf_path}")
    
    target_tags = list(app_config.TAG_DESCRIPTIONS.keys())
    print(f"[JOB-{job.job_id}] TAG_DESCRIPTIONS has {len(app_config.TAG_DESCRIPTIONS)} tags: {target_tags}")
    
    await job.broadcast({"type": "start", "message": "Starting extraction..."})
    
    results = []
    fitz = get_fitz()
    
    try:
        start_time = time.time()
        print(f"[JOB-{job.job_id}] Opening PDF...")
        
        # Use PyMuPDF for faster PDF opening
        if is_pymupdf_available():
            pdf_doc = fitz.open(job.pdf_path)
            pdf_open_time = time.time() - start_time
            print(f"[JOB-{job.job_id}] PDF opened with PyMuPDF in {pdf_open_time:.2f}s")
            use_pymupdf = True
        else:
            import pdfplumber
            pdf_doc = pdfplumber.open(job.pdf_path)
            pdf_open_time = time.time() - start_time
            print(f"[JOB-{job.job_id}] PDF opened with pdfplumber in {pdf_open_time:.2f}s")
            use_pymupdf = False
        
        try:
            total = min(job.end_page, len(pdf_doc)) - job.start_page + 1
            job.total_pages = total
            
            await job.broadcast({"type": "info", "total_pages": total})
            
            # Store ALL region data across ALL pages for position-based deduplication
            all_regions_data = []
            
            for idx, page_num in enumerate(range(job.start_page, min(job.end_page + 1, len(pdf_doc) + 1))):
                job.current_page = page_num
                
                print(f"[JOB-{job.job_id}] Starting page {page_num} (subscribers: {len(job.subscribers)})")
                
                # Get page
                def get_page():
                    if use_pymupdf:
                        return pdf_doc[page_num - 1]
                    else:
                        return pdf_doc.pages[page_num - 1]
                
                page = await asyncio.to_thread(get_page)
                sheet = "N/A"
                print(f"[JOB-{job.job_id}] Page {page_num} loaded")
                
                # Send page start
                await job.broadcast({
                    "type": "page_start",
                    "page": page_num,
                    "sheet": sheet,
                    "progress": idx + 1,
                    "total": total
                })
                
                # Initialize page tracking
                job.processed_pages[page_num] = {
                    'sheet': sheet,
                    'full_page_url': None,
                    'drawings': [],
                    'total_drawings': 0
                }
                job.current_drawing_index = 0
                
                # Step 1: Render page to image
                print(f"[JOB-{job.job_id}] Rendering page {page_num}")
                try:
                    img_original, resolution = await asyncio.to_thread(
                        render_page_to_image, page, page_num, use_pymupdf
                    )
                    print(f"[JOB-{job.job_id}] Page rendered successfully")
                except Exception as e:
                    print(f"[JOB-{job.job_id}] Render ERROR: {type(e).__name__}: {e}")
                    img_original, resolution = None, None
                
                # Step 2: Send original image immediately
                if img_original:
                    original_resized = img_original.copy()
                    if original_resized.width > 1400:
                        ratio = 1400 / original_resized.width
                        new_height = int(original_resized.height * ratio)
                        original_resized = original_resized.resize((1400, new_height), Image.Resampling.LANCZOS)
                    
                    original_filename = f"{job.job_id}-page-{page_num}-original.jpg"
                    original_url = save_image_to_disk(original_resized, original_filename, quality=75)
                    print(f"[JOB-{job.job_id}] Saved original page to {original_url}")
                    
                    await job.broadcast({
                        "type": "full_page",
                        "image_url": original_url,
                        "page": page_num,
                        "sheet": sheet,
                        "drawing_count": 0,
                        "annotated": False
                    })
                
                # Step 3: Detect drawings using Roboflow
                if img_original and resolution:
                    print(f"[JOB-{job.job_id}] Starting detection for page {page_num}")
                    try:
                        img_annotated, cropped_drawings, drawing_data = await asyncio.to_thread(
                            detect_drawings_on_image, img_original, page_num, resolution
                        )
                        print(f"[JOB-{job.job_id}] Detection complete: {len(cropped_drawings) if cropped_drawings else 0} drawings")
                    except Exception as e:
                        print(f"[JOB-{job.job_id}] Detection ERROR: {type(e).__name__}: {e}")
                        img_annotated, cropped_drawings, drawing_data = None, [], []
                else:
                    img_annotated, cropped_drawings, drawing_data = None, [], []
                
                # Step 4: Send annotated image
                if img_annotated:
                    annotated_resized = img_annotated.copy()
                    if annotated_resized.width > 1400:
                        ratio = 1400 / annotated_resized.width
                        new_height = int(annotated_resized.height * ratio)
                        annotated_resized = annotated_resized.resize((1400, new_height), Image.Resampling.LANCZOS)
                    
                    annotated_filename = f"{job.job_id}-page-{page_num}-full.jpg"
                    annotated_url = save_image_to_disk(annotated_resized, annotated_filename, quality=75)
                    print(f"[JOB-{job.job_id}] Saved annotated page to {annotated_url}")
                    
                    job.processed_pages[page_num]['full_page_url'] = annotated_url
                    job.processed_pages[page_num]['total_drawings'] = len(cropped_drawings)
                    
                    await job.broadcast({
                        "type": "full_page",
                        "image_url": annotated_url,
                        "page": page_num,
                        "sheet": sheet,
                        "drawing_count": len(cropped_drawings),
                        "annotated": True
                    })
                
                if cropped_drawings:
                    num_drawings = len(cropped_drawings)
                    page_has_valid_tags = False
                    
                    # Process each drawing
                    for i, crop_data in enumerate(cropped_drawings):
                        job.current_drawing_index = i
                        
                        cropped_img = crop_data["image"]
                        conf = crop_data["confidence"]
                        
                        # Save drawing image
                        drawing_img = cropped_img.copy()
                        if drawing_img.width > 1200 or drawing_img.height > 1200:
                            drawing_img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                        
                        drawing_filename = f"{job.job_id}-page-{page_num}-drawing-{i+1}.jpg"
                        drawing_url = save_image_to_disk(drawing_img, drawing_filename, quality=80)
                        
                        job.processed_pages[page_num]['drawings'].append({
                            'index': i + 1,
                            'image_url': drawing_url,
                            'confidence': f"{conf:.0%}",
                            'bbox': crop_data["bbox"]
                        })
                        
                        await job.broadcast({
                            "type": "drawing",
                            "index": i + 1,
                            "total_drawings": num_drawings,
                            "image_url": drawing_url,
                            "confidence": f"{conf:.0%}",
                            "bbox": crop_data["bbox"]
                        })
                        
                        # Step 1: Try PyMuPDF first, then OCR fallback
                        text = ""
                        word_positions = []
                        mupdf_rect = None
                        is_ocr_text = False
                        
                        try:
                            if is_pymupdf_available() and img_original and resolution:
                                if use_pymupdf and is_pymupdf_available():
                                    page_rect = page.rect
                                    page_width = page_rect.width
                                    page_height = page_rect.height
                                else:
                                    page_width, page_height = page.width, page.height
                                
                                image_size = img_original.size
                                pdf_size = (page_width, page_height)
                                image_bbox = crop_data["bbox"]
                                
                                pdf_bbox = image_coords_to_pdf_coords(image_bbox, image_size, pdf_size, resolution)
                                
                                def extract_pdf_text():
                                    return extract_text_from_pdf_region(job.pdf_path, page_num, pdf_bbox, pdf_size, return_positions=True)
                                
                                extraction_result = await asyncio.to_thread(extract_pdf_text)
                                text, word_positions, mupdf_rect = extraction_result
                                
                                if text:
                                    print(f"[JOB-{job.job_id}] PyMuPDF extracted {len(text)} chars from PDF")
                                    is_ocr_text = False
                                else:
                                    if ALWAYS_PYMUPDF:
                                        print(f"[JOB-{job.job_id}] PyMuPDF: No selectable text found, OCR disabled")
                                    else:
                                        print(f"[JOB-{job.job_id}] PyMuPDF: No text found, falling back to OCR")
                            
                            # Fallback to OCR
                            if not text and not ALWAYS_PYMUPDF:
                                print(f"[JOB-{job.job_id}] Starting OCR for drawing {i+1}")
                                ocr_result = await asyncio.to_thread(ocr_image, cropped_img)
                                text = ocr_result[0] if isinstance(ocr_result, tuple) else ocr_result
                                print(f"[JOB-{job.job_id}] OCR complete: {len(text)} chars")
                                is_ocr_text = True
                            
                        except Exception as e:
                            print(f"[JOB-{job.job_id}] Text extraction ERROR: {type(e).__name__}: {e}")
                            import traceback
                            print(traceback.format_exc())
                            text = ""
                            is_ocr_text = False
        
                        # Step 2: Check for tags
                        location_desc = ""
                        
                        matched_tags, drawing_results = await asyncio.to_thread(
                            process_ocr_and_tags, text, conf, sheet, page_num, "", is_ocr_text, target_tags, app_config.TAG_DESCRIPTIONS
                        )
                        
                        # Step 3: Run OpenAI ONLY if tags found
                        valid_tags_exist = any(not r.get("_debug_ignored") for r in drawing_results)
                        
                        if valid_tags_exist:
                            page_has_valid_tags = True
                        
                        drawing_id = ""
                        if valid_tags_exist and is_openai_available():
                            try:
                                print(f"[JOB-{job.job_id}] Starting OpenAI call for drawing {i+1}")
                                drawing_id, location_desc = await asyncio.to_thread(extract_location_description, cropped_img)
                                print(f"[JOB-{job.job_id}] OpenAI call complete: ID={drawing_id}, LOC={location_desc[:30] if location_desc else ''}")
                            except Exception as e:
                                print(f"[JOB-{job.job_id}] OpenAI ERROR: {type(e).__name__}: {e}")
                                drawing_id, location_desc = "", ""
                        elif not valid_tags_exist:
                             drawing_id, location_desc = "", ""

                        # Store region data for position-based deduplication
                        all_regions_data.append({
                            'drawing_index': i + 1,
                            'word_positions': word_positions,
                            'mupdf_rect': mupdf_rect,
                            'sheet': sheet,
                            'page': page_num,
                            'location': location_desc,
                            'drawing_id': drawing_id,
                            'confidence': conf,
                            'bbox': crop_data["bbox"]
                        })

                        # Update results with location
                        for r in drawing_results:
                            r['location'] = location_desc
                            r['description'] = drawing_id
                            r['drawing_index'] = i + 1
                            r['bbox'] = crop_data["bbox"]
                            r['word_positions'] = word_positions
                            r['mupdf_rect'] = mupdf_rect
                            
                        # Process results - collect valid tags for clean summary
                        valid_results = []
                        found_tag_names = []
                        for result in drawing_results:
                            if result.get("_debug_ignored"):
                                # Debug: OCR saw something but no match - server log only
                                print(f"[JOB-{job.job_id}] OCR saw '{result['tag']}' but matched none of {len(target_tags)} tags")
                            elif result.get("_debug_match"):
                                del result["_debug_match"]
                                valid_results.append(result)
                                found_tag_names.append(result['tag'])
                                
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
                                if result.get('tag'):
                                    found_tag_names.append(result['tag'])
                        
                        # Send ONE clean log per drawing (same as app_ref.py)
                        if found_tag_names:
                            tags_str = ", ".join(found_tag_names)
                            await job.broadcast({"type": "log", "level": "success", "message": f"Drawing {i+1}: Found tags ({tags_str})"})
                        else:
                            await job.broadcast({"type": "log", "level": "info", "message": f"Drawing {i+1}: No tags"})
                        
                        if valid_results:
                            job.results.extend(valid_results)
                        
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
                        job.results = results
                    
                    # Sheet name extraction (only when desired tags found on this page)
                    if page_has_valid_tags and is_openai_available() and img_original:
                        try:
                            print(f"[JOB-{job.job_id}] USING OPENAI for sheet name on page {page_num}")
                            openai_sheet = await asyncio.to_thread(extract_sheet_name_openai, img_original)
                            
                            if openai_sheet:
                                sheet = openai_sheet
                                print(f"[JOB-{job.job_id}] OpenAI sheet name for page {page_num}: {sheet}")
                            else:
                                # OpenAI returned empty - fall back to regex extraction
                                print(f"[JOB-{job.job_id}] OpenAI couldn't extract sheet name, falling back to regex")
                                sheet = await asyncio.to_thread(get_sheet_number, page, use_pymupdf)
                                print(f"[JOB-{job.job_id}] Regex fallback sheet name: {sheet}")
                        except Exception as e:
                            # OpenAI failed - fall back to regex extraction
                            print(f"[JOB-{job.job_id}] Sheet name extraction ERROR: {type(e).__name__}: {e}")
                            sheet = await asyncio.to_thread(get_sheet_number, page, use_pymupdf)
                            print(f"[JOB-{job.job_id}] Regex fallback sheet name: {sheet}")
                    elif page_has_valid_tags and not is_openai_available():
                        # OpenAI not available at all - fall back to regex
                        print(f"[JOB-{job.job_id}] OpenAI not available, falling back to regex")
                        sheet = await asyncio.to_thread(get_sheet_number, page, use_pymupdf)
                        print(f"[JOB-{job.job_id}] Regex fallback sheet name: {sheet}")
                    
                    # Update all data with final sheet name
                    if page_has_valid_tags and sheet != "N/A":
                        for region in all_regions_data:
                            if region['page'] == page_num:
                                region['sheet'] = sheet
                        
                        if page_num in job.processed_pages:
                            job.processed_pages[page_num]['sheet'] = sheet
                        
                        # Send sheet name update to UI (separate event to avoid incrementing page counter)
                        await job.broadcast({
                            "type": "sheet_update",
                            "page": page_num,
                            "sheet": sheet
                        })
                
                # Release memory
                del page
                if img_original:
                    del img_original
                if img_annotated:
                    del img_annotated
                if cropped_drawings:
                    del cropped_drawings
                if drawing_data:
                    del drawing_data
                
                gc.collect()
                await asyncio.sleep(0.1)
            
            # Position-based deduplication — PER TAG
            # Each tag is deduplicated independently so nearby different tags
            # (e.g. A_WB-3 and A_PT-3 in same ROOM FINISH table) don't interfere.
            all_detections = []
            
            print(f"[JOB-{job.job_id}] Searching {len(all_regions_data)} regions for {len(target_tags)} target tags...")
            
            for region_data in all_regions_data:
                word_positions = region_data['word_positions']
                mupdf_rect = region_data['mupdf_rect']
                drawing_idx = region_data['drawing_index']
                
                if not word_positions:
                    continue
                
                for target_tag in target_tags:
                    tag_positions = find_all_tag_positions_cleaned(target_tag, word_positions)
                    
                    for pos in tag_positions:
                        all_detections.append({
                            'result': {
                                'material_type': app_config.TAG_DESCRIPTIONS.get(target_tag, "Unknown"),
                                'tag': target_tag,
                                'sheet': region_data['sheet'],
                                'page': region_data['page'],
                                'description': region_data['drawing_id'],
                                'location': region_data['location'],
                                'confidence': f"{region_data['confidence']:.0%}",
                                'drawing_index': drawing_idx
                            },
                            'position': pos,
                            'mupdf_rect': mupdf_rect,
                            'drawing_index': drawing_idx,
                            'tag_key': (target_tag, region_data['sheet'], region_data['page'])
                        })
            
            print(f"[JOB-{job.job_id}] Total raw detections: {len(all_detections)}")
            
            # Group detections BY TAG first, then dedup positions within each tag
            from collections import defaultdict as _defaultdict
            detections_by_tag = _defaultdict(list)
            for det in all_detections:
                detections_by_tag[det['result']['tag']].append(det)
            
            final_results = []
            total_dupes = 0
            
            for tag_name, tag_detections in detections_by_tag.items():
                # Dedup positions within this tag only
                unique_positions = []
                
                for detection in tag_detections:
                    pos = detection['position']
                    
                    found_match = False
                    for unique in unique_positions:
                        if positions_are_same(pos, unique['position'], tolerance=5):
                            unique['detections'].append(detection)
                            found_match = True
                            break
                    
                    if not found_match:
                        unique_positions.append({
                            'position': pos,
                            'detections': [detection]
                        })
                
                dupes = len(tag_detections) - len(unique_positions)
                total_dupes += dupes
                
                # For each unique position of this tag, pick the best detection
                for unique in unique_positions:
                    tag_pos = unique['position']
                    dets = unique['detections']
                    
                    if len(dets) == 1:
                        final_results.append(dets[0]['result'])
                    else:
                        containing_detections = [
                            d for d in dets
                            if d['mupdf_rect'] and is_point_inside_bbox(tag_pos, d['mupdf_rect'])
                        ]
                        
                        if len(containing_detections) == 0:
                            print(f"[JOB-{job.job_id}] WARNING: Tag {tag_name} at {tag_pos} not inside any mupdf_rect")
                            final_results.append(dets[0]['result'])
                        elif len(containing_detections) == 1:
                            final_results.append(containing_detections[0]['result'])
                        else:
                            best_detection = containing_detections[0]
                            min_distance = distance_to_region_center(tag_pos, best_detection['mupdf_rect'])
                            for d in containing_detections[1:]:
                                dist = distance_to_region_center(tag_pos, d['mupdf_rect'])
                                if dist < min_distance:
                                    min_distance = dist
                                    best_detection = d
                            print(f"[JOB-{job.job_id}] Tag {tag_name} in multiple regions, assigned to drawing {best_detection['drawing_index']}")
                            final_results.append(best_detection['result'])
            
            print(f"[JOB-{job.job_id}] Unique positions: {len(final_results)}, duplicates removed: {total_dupes}")
            
            print(f"[JOB-{job.job_id}] Final results: {len(final_results)} unique tag(s)")
            
            # Save results
            csv_path = os.path.join(RESULTS_FOLDER, "results.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
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
            
            # Job completed
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            
            await job.broadcast({
                "type": "complete",
                "total_tags": len(final_results),
                "results": final_results,
                "job_id": job.job_id
            })
            
            print(f"[JOB-{job.job_id}] COMPLETED - {len(final_results)} unique tags found")
            
        finally:
            if use_pymupdf and is_pymupdf_available():
                pdf_doc.close()
            elif not use_pymupdf:
                pdf_doc.close()
            
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        print(f"[JOB-{job.job_id}] FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        await job.broadcast({"type": "error", "message": str(e)})
    
    # Auto-cleanup
    if AUTO_CLEANUP:
        await asyncio.sleep(0.5)
        cleanup_temp_files()
        await job.broadcast({
            "type": "log",
            "level": "info",
            "message": "Auto-cleanup: Temporary files deleted (CSV results kept)"
        })
    
    # Cleanup job resources after delay
    await asyncio.sleep(305)
    cleanup_job_resources(job.job_id)
