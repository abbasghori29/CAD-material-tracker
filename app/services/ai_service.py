"""
AI service - OpenAI Vision and Roboflow integration.
"""

import os
import base64
import tempfile
from typing import List, Optional, Tuple
from io import BytesIO
from PIL import Image

from app.core.config import OPENAI_API_KEY, ROBOFLOW_API_KEY, ROBOFLOW_MODEL_ID
from app.core.constants import LOCATION_EXTRACTION_PROMPT, SHEET_NAME_EXTRACTION_PROMPT
from app.models.extraction import DrawingLocationInfo, SheetNameInfo, BatchLocationInfo

# Service availability flags
OPENAI_AVAILABLE = bool(OPENAI_API_KEY)
ROBOFLOW_AVAILABLE = False
VISION_LLM = None
ROBOFLOW_CLIENT = None

# Initialize OpenAI Vision
if OPENAI_AVAILABLE:
    try:
        from langchain_openai import ChatOpenAI
        VISION_LLM = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=OPENAI_API_KEY,
            timeout=60,
            max_retries=2
        )
        print("[AI] OpenAI Vision enabled with gpt-4o model")
    except Exception as e:
        print(f"[AI] WARNING: Failed to initialize OpenAI: {e}")
        OPENAI_AVAILABLE = False

# Initialize Roboflow
try:
    import importlib
    inference_sdk = importlib.import_module('inference_sdk')
    InferenceHTTPClient = getattr(inference_sdk, 'InferenceHTTPClient')
    ROBOFLOW_CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )
    ROBOFLOW_AVAILABLE = True
    print("[AI] Roboflow client initialized")
except ImportError:
    print("[AI] Roboflow inference_sdk not installed")
except Exception as e:
    print(f"[AI] Roboflow initialization error: {e}")


def _image_to_base64(img: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    img.save(buffered, format=format, quality=quality)
    return base64.b64encode(buffered.getvalue()).decode()


def extract_location_description(img: Image.Image) -> Tuple[str, str]:
    """
    Extract location description from CAD drawing using OpenAI Vision (single image).
    
    Args:
        img: PIL Image of the drawing
        
    Returns:
        Tuple of (drawing_id, location_description)
    """
    if not OPENAI_AVAILABLE or VISION_LLM is None:
        return "", ""
    
    try:
        from langchain_core.messages import HumanMessage
        
        # Convert image to base64
        img_b64 = _image_to_base64(img)
        
        # Create message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": LOCATION_EXTRACTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        )
        
        # Use structured output
        structured_llm = VISION_LLM.with_structured_output(DrawingLocationInfo)
        result = structured_llm.invoke([message])
        
        return result.drawing_id, result.location_description
        
    except Exception as e:
        print(f"[AI] Location extraction error: {e}")
        return "", ""


def extract_sheet_name_openai(img: Image.Image) -> str:
    """
    Extract sheet name/number from a CAD drawing page using OpenAI Vision.
    
    Only call this when desired tags have been found on the page.
    Sends the full page image to identify the sheet number from the title block.
    
    Args:
        img: Full page image (PIL Image)
    
    Returns:
        str: Extracted sheet name, or empty string if not found
    """
    if not OPENAI_AVAILABLE or VISION_LLM is None:
        return ""
    
    try:
        from langchain_core.messages import HumanMessage
        
        # Reduce image size for faster processing (sheet name is usually in title block)
        max_size = 1500
        if img.width > max_size or img.height > max_size:
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        img_b64 = _image_to_base64(img, quality=70)
        
        # Create message
        message = HumanMessage(
            content=[
                {"type": "text", "text": SHEET_NAME_EXTRACTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        )
        
        # Use structured output
        structured_llm = VISION_LLM.with_structured_output(SheetNameInfo)
        result = structured_llm.invoke([message])
        
        return result.sheet_name if result else ""
        
    except Exception as e:
        print(f"[AI] Sheet name extraction error: {e}")
        return ""


def extract_locations_batch(images: List[Image.Image]) -> List[str]:
    """
    Extract location descriptions from MULTIPLE drawings in a SINGLE API call.
    
    Args:
        images: List of PIL Images (cropped drawings)
        
    Returns:
        List of location descriptions (one per image)
    """
    if not OPENAI_AVAILABLE or VISION_LLM is None:
        return [""] * len(images)
    
    if not images:
        return []
    
    try:
        from langchain_core.messages import HumanMessage
        
        # Build content with all images
        content = [
            {"type": "text", "text": f"""You are analyzing {len(images)} CAD drawing images to extract location/title from each.

For each drawing, look for:
- Title block text (usually bottom or right side)
- Drawing title/label (large text near top)
- Location identifier (room names, elevation labels, section names)

Return a list of {len(images)} location descriptions, one for each drawing in order.
If you can't find a location for a drawing, use empty string."""}
        ]
        
        # Add each image
        for i, img in enumerate(images):
            # Resize for efficiency
            max_size = 800
            if img.width > max_size or img.height > max_size:
                ratio = min(max_size / img.width, max_size / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            img_b64 = _image_to_base64(img, quality=70)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        
        message = HumanMessage(content=content)
        
        # Use structured output
        structured_llm = VISION_LLM.with_structured_output(BatchLocationInfo)
        result = structured_llm.invoke([message])
        
        if result and result.locations:
            # Pad or trim to match input length
            locations = result.locations[:len(images)]
            while len(locations) < len(images):
                locations.append("")
            return locations
        
        return [""] * len(images)
        
    except Exception as e:
        print(f"[AI] Batch location extraction error: {e}")
        return [""] * len(images)


def get_roboflow_client():
    """Get the Roboflow client instance"""
    return ROBOFLOW_CLIENT


def is_roboflow_available() -> bool:
    """Check if Roboflow is available"""
    return ROBOFLOW_AVAILABLE


def is_openai_available() -> bool:
    """Check if OpenAI is available"""
    return OPENAI_AVAILABLE
