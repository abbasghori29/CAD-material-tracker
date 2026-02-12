"""
OCR service - Multi-engine OCR with fallback support.
Supports LlamaParse, EasyOCR, and Tesseract.
"""

import os
import tempfile
from typing import Optional, Tuple
from PIL import Image
import numpy as np

from app.core.config import LLAMA_PARSE_API_KEY, ALWAYS_PYMUPDF

# OCR Engine availability flags
LLAMA_PARSE = None
EASY_OCR_READER = None
TESSERACT_AVAILABLE = False

# Initialize LlamaParse - BEST for CAD drawings
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

# Initialize EasyOCR - Fallback option
try:
    import easyocr
    EASY_OCR_READER = easyocr.Reader(['en'], gpu=False)
    print("[OCR] EasyOCR initialized as fallback")
except (ImportError, OSError) as e:
    print(f"[OCR] EasyOCR not available: {e}")

# Initialize Tesseract - Last resort fallback
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    print("[OCR] Tesseract available as last resort")
except ImportError:
    pass


def ocr_image(img: Image.Image) -> Tuple[str, float]:
    """
    OCR an image - tries LlamaParse (best), then EasyOCR, then Tesseract.
    
    Args:
        img: PIL Image to process
        
    Returns:
        Tuple of (extracted_text, confidence_score)
    """
    # If ALWAYS_PYMUPDF is set, skip OCR entirely (use PDF text extraction instead)
    if ALWAYS_PYMUPDF:
        return "", 0.0
    
    text = ""
    confidence = 0.0
    
    # Try LlamaParse first (best for CAD)
    if LLAMA_PARSE is not None:
        try:
            # Save image temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name, format='PNG')
                tmp_path = tmp.name
            
            # Parse with LlamaParse
            documents = LLAMA_PARSE.load_data(tmp_path)
            if documents:
                text = " ".join([doc.text for doc in documents])
                confidence = 0.95  # LlamaParse is very accurate
            
            # Cleanup temp file
            os.unlink(tmp_path)
            
            if text.strip():
                return text, confidence
        except Exception as e:
            print(f"[OCR] LlamaParse error: {e}")
    
    # Try EasyOCR second
    if EASY_OCR_READER is not None:
        try:
            # Convert PIL image to numpy array
            img_array = np.array(img)
            results = EASY_OCR_READER.readtext(img_array)
            
            if results:
                texts = []
                confidences = []
                for bbox, detected_text, conf in results:
                    texts.append(detected_text)
                    confidences.append(conf)
                
                text = " ".join(texts)
                confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            if text.strip():
                return text, confidence
        except Exception as e:
            print(f"[OCR] EasyOCR error: {e}")
    
    # Try Tesseract as last resort
    if TESSERACT_AVAILABLE:
        try:
            import pytesseract
            text = pytesseract.image_to_string(img)
            confidence = 0.7  # Tesseract is generally less accurate for CAD
            
            if text.strip():
                return text, confidence
        except Exception as e:
            print(f"[OCR] Tesseract error: {e}")
    
    return text, confidence


def is_ocr_available() -> bool:
    """Check if any OCR engine is available"""
    return LLAMA_PARSE is not None or EASY_OCR_READER is not None or TESSERACT_AVAILABLE
