"""
Text extractor - PDF text extraction and coordinate conversion.
"""

import re
from typing import Optional, Tuple, List

# PyMuPDF availability
PYMUPDF_AVAILABLE = False
fitz = None

try:
    import fitz as _fitz
    fitz = _fitz
    PYMUPDF_AVAILABLE = True
    print("[TEXT] PyMuPDF available - will try PDF text extraction first")
except ImportError:
    print("[TEXT] PyMuPDF not installed - install with: pip install pymupdf")


def image_coords_to_pdf_coords(image_bbox, image_size, pdf_size, resolution):
    """
    Convert image pixel coordinates to PDF coordinates.
    
    PDF coordinate system: origin at bottom-left, y increases upward
    Image coordinate system: origin at top-left, y increases downward
    
    Args:
        image_bbox: (x1, y1, x2, y2) in image pixels (top-left origin)
        image_size: (width, height) in pixels
        pdf_size: (width, height) in PDF points
        resolution: DPI used to render the image
    
    Returns:
        (x1, y1, x2, y2) in PDF coordinates (bottom-left origin)
    """
    img_width, img_height = image_size
    pdf_width, pdf_height = pdf_size
    
    x1_img, y1_img, x2_img, y2_img = image_bbox
    
    # Scale factor: image pixels to PDF units
    # resolution = pixels per inch, PDF usually 72 points per inch
    scale = 72.0 / resolution
    
    # Convert x coordinates (straightforward scaling)
    x1_pdf = x1_img * scale
    x2_pdf = x2_img * scale
    
    # Convert y coordinates (flip because PDF y goes up, image y goes down)
    y1_pdf = pdf_height - (y2_img * scale)  # Bottom of box in PDF
    y2_pdf = pdf_height - (y1_img * scale)  # Top of box in PDF
    
    return (x1_pdf, y1_pdf, x2_pdf, y2_pdf)


def extract_text_from_pdf_region(
    pdf_path: str, 
    page_num: int, 
    pdf_bbox, 
    pdf_size, 
    return_positions: bool = False
):
    """
    Extract selectable text from a PDF region using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        pdf_bbox: (x1, y1, x2, y2) in PDF coordinates (bottom-left origin)
        pdf_size: (width, height) in PDF points
        return_positions: If True, also returns word positions for tag location tracking
    
    Returns:
        If return_positions=False: Extracted text string, or empty string if no text or error
        If return_positions=True: Tuple of (text, word_positions, mupdf_rect) where:
            - word_positions is list of (word, x, y) with PyMuPDF coordinates
            - mupdf_rect is the actual clipping rect (for containment checks)
    """
    if not PYMUPDF_AVAILABLE:
        return ("", [], None) if return_positions else ""

    x1, y1, x2, y2 = pdf_bbox
    pdf_width, pdf_height = pdf_size
    
    # PyMuPDF uses top-left origin, convert from bottom-left PDF coords
    y1_mupdf = pdf_height - y2  # Top edge
    y2_mupdf = pdf_height - y1  # Bottom edge
    
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]  # 0-indexed
        
        # Ensure coordinates are within page bounds
        x1 = max(0, min(x1, pdf_width))
        y1_mupdf = max(0, min(y1_mupdf, pdf_height))
        x2 = max(0, min(x2, pdf_width))
        y2_mupdf = max(0, min(y2_mupdf, pdf_height))
        
        # Create a rectangle for the region
        rect = fitz.Rect(x1, y1_mupdf, x2, y2_mupdf)
        
        # Store mupdf_rect for containment checking
        mupdf_rect = (x1, y1_mupdf, x2, y2_mupdf)
        
        # Extract text from rectangle
        text = page.get_text("text", clip=rect)
        
        # Also get word positions if requested
        word_positions = []
        if return_positions:
            words = page.get_text("words", clip=rect)
            # words format: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            for word_data in words:
                word_text = word_data[4]
                # Use center of word for position
                word_x = (word_data[0] + word_data[2]) / 2
                word_y = (word_data[1] + word_data[3]) / 2
                word_positions.append((word_text, word_x, word_y))
        
        doc.close()
        
        if return_positions:
            return text.strip(), word_positions, mupdf_rect
        return text.strip()
        
    except Exception as e:
        print(f"[TEXT] Error extracting text from region: {e}")
        if return_positions:
            return "", [], None
        return ""


def get_sheet_number(page, use_pymupdf: bool = False) -> str:
    """
    Extract sheet number - looks for sheet number near title block keywords
    
    Args:
        page: Either a PyMuPDF page (fitz.Page) or pdfplumber page
        use_pymupdf: If True, expects PyMuPDF page, else pdfplumber page
    
    Returns:
        Extracted sheet number or empty string
    """
    try:
        if use_pymupdf and PYMUPDF_AVAILABLE:
            # PyMuPDF extraction
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height
            
            # Focus on title block area (bottom-right quadrant typically)
            # Also check right side for rotated/different layouts
            search_regions = [
                # Bottom-right (most common)
                fitz.Rect(page_width * 0.5, page_height * 0.7, page_width, page_height),
                # Right side
                fitz.Rect(page_width * 0.75, 0, page_width, page_height),
                # Bottom
                fitz.Rect(0, page_height * 0.85, page_width, page_height),
            ]
            
            all_text = ""
            for region in search_regions:
                text = page.get_text("text", clip=region)
                all_text += text + "\n"
        else:
            # pdfplumber extraction
            page_width = page.width
            page_height = page.height
            
            # Focus on title block area
            title_block_bbox = (
                page_width * 0.5,  # x0
                page_height * 0.7,  # y0
                page_width,  # x1
                page_height  # y1
            )
            
            cropped = page.within_bbox(title_block_bbox)
            all_text = cropped.extract_text() or ""
        
        # Sheet number patterns - look for common formats
        sheet_patterns = [
            # Standard: A-101, E-201, etc.
            r'\b([AESMPCILGD][-\s]?\d{1,3}(?:\.\d{1,2})?)\b',
            # With prefix: SHEET A-101
            r'SHEET\s*[:#]?\s*([A-Z][-\s]?\d{1,3})',
            # Decimal: A2.01
            r'\b([A-Z]\d\.\d{2})\b',
            # Just numbers with prefix: A101
            r'\b([AESMPCILGD]\d{3})\b',
        ]
        
        for pattern in sheet_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                # Return first valid match
                sheet = matches[0].upper().strip()
                # Validate - should have at least one letter and one digit
                if re.search(r'[A-Z]', sheet) and re.search(r'\d', sheet):
                    return sheet
        
        return ""
        
    except Exception as e:
        print(f"[TEXT] Error getting sheet number: {e}")
        return ""


def is_pymupdf_available() -> bool:
    """Check if PyMuPDF is available"""
    return PYMUPDF_AVAILABLE


def get_fitz():
    """Get the fitz (PyMuPDF) module"""
    return fitz
