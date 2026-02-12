"""
Extraction models - Pydantic schemas for AI extraction results.
"""

from typing import List
from pydantic import BaseModel, Field


class DrawingLocationInfo(BaseModel):
    """Extracted location information from a CAD drawing."""
    drawing_id: str = Field(
        description="The alphanumeric code/identifier for the drawing, usually found to the left of the title or in a circle. Examples: '1/A101', '5', 'A-2'. Return empty string if not found."
    )
    location_description: str = Field(
        description="The location identifier/description found in the drawing title block or header. Examples: 'E1 1/8 RCP - LEVEL 3 PART B', 'C3 KITCHEN E1 ELEVATION 2', 'B1 SECTION AT EAST OF GARAGE - NS'. Return empty string if not found."
    )


class SheetNameInfo(BaseModel):
    """Extracted sheet name/number from a CAD drawing page."""
    sheet_name: str = Field(
        description="The sheet number/name found in the title block. Examples: 'A-101', 'I-700', 'E-201', 'A2.01', 'S-301.1', 'M-100'. Return empty string if not found."
    )


class BatchLocationInfo(BaseModel):
    """Batch extraction of locations from multiple drawings."""
    locations: List[str] = Field(
        description="List of location descriptions for each drawing, in order. Use empty string if not found."
    )
