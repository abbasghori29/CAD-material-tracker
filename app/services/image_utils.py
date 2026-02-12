"""
Image utilities - Functions for image conversion, storage, and manipulation.
"""

import os
import base64
import glob
from io import BytesIO
from PIL import Image, ImageFile

from app.core import config as app_config

# Prevent PIL crash on truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 500_000_000


def image_to_base64(img: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    img.save(buffered, format=format, quality=quality)
    return base64.b64encode(buffered.getvalue()).decode()


def save_image_to_disk(img: Image.Image, filename: str, quality: int = 80) -> str:
    """Save image to disk and return URL path"""
    filepath = os.path.join(app_config.IMAGES_FOLDER, filename)
    img.save(filepath, "JPEG", quality=quality)
    return f"/output_images/{filename}"


def cleanup_job_images(job_id: str):
    """Delete all images for a job"""
    pattern = os.path.join(app_config.IMAGES_FOLDER, f"{job_id}-*")
    for filepath in glob.glob(pattern):
        try:
            os.remove(filepath)
        except:
            pass
