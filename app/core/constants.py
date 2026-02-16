"""
Constants and patterns used across the application.
"""

import re

# === TAG PATTERN ===
# Regex for tags - FLEXIBLE to handle any length and common CAD separators
# Common CAD separators: hyphen (-), underscore (_), period (.), slash (/)
# Pattern matches: LETTERS + optional separator + optional more LETTERS + separator + DIGIT-LIKE + optional suffix
# Examples: BR-1, A_WC-12, A_PT-1A, APT-IA (1→I OCR error), WC-12
# Requires at least one digit-like character: 0-9, I, l, O (common OCR confusions)
TAG_PATTERN = re.compile(r'(?i)[A-Z]+[-_./]?[A-Z]*[-_./]?[0-9IlO][A-Z0-9IlO]*')

# === PROMPTS ===
# Shared prompt for location extraction
LOCATION_EXTRACTION_PROMPT = """Extract the drawing identifier and title/location from this CAD drawing image.

Look carefully at these areas:
- The BOLD text at the bottom-left of the drawing (contains the drawing ID, usually in large bold font).
- The circular bubble or rectangular box usually to the left of or above the main title.
- Title block (usually bottom-right corner or top of page).
- Drawing header/label (large text near top).

Extraction Rules:
1. Drawing ID: Extract the alphanumeric code that identifies this specific drawing. It is typically bold text at the bottom of the drawing, to the left of the title.
   - Drawing IDs are very commonly a LETTER followed by a NUMBER, e.g.: 'B7', 'A3', 'C1', 'D2', 'E5'
   - Other valid formats: '1/A101', 'A', '1', '2 ID202', '1-A101'
   - IMPORTANT: Pay close attention to distinguish letters from numbers. 'B' is NOT '8', '7' is NOT '1', 'I' is NOT '1'. Read the characters carefully.
   - If you see bold text like 'B7' at the bottom-left, that is the drawing ID, not '1'.
2. Location: Extract the main title or location identifier (e.g., 'PRESENTATION ELEVATION - KID\'S HIDEOUT', 'LEVEL 3 PLAN', 'SECTION AT GARAGE').

Return both fields. Use empty string if a field is not found."""

# Sheet name extraction prompt
SHEET_NAME_EXTRACTION_PROMPT = """You are analyzing a CAD/architectural drawing page to extract the SHEET NUMBER from the title block.

IMPORTANT RULES:
1. Look at the TITLE BLOCK area (usually bottom-right corner of the page).
2. Find the SHEET NUMBER which typically follows a pattern like:
   - Letter prefix + numbers: A-101, E-201, M-100, S-301
   - With decimals: A2.01, S-301.1
   - With letters in numbers: I-700 (where I is the letter, not number 1)
3. The sheet number is usually prominently displayed and may be near labels like "SHEET", "DWG NO", "DRAWING NO".
4. Common prefixes: A=Architectural, E=Electrical, S=Structural, M=Mechanical, P=Plumbing, I=Interior, C=Civil
5. DO NOT return:
   - Dates or revision numbers
   - Project numbers (long numbers like 12345 or dates)
6. Return ONLY the sheet number/name. Return empty string if you cannot confidently identify one.
7. The sheet number is usually one of the most prominent pieces of text in the title block."""

# === OCR SUBSTITUTION PATTERNS ===
# Common OCR misreadings for tag matching
OCR_SUBSTITUTIONS = {
    '0': ['O', 'o', 'Q', 'D'],
    'O': ['0', 'o', 'Q', 'D'],
    '1': ['I', 'i', 'l', '|', '!'],
    'I': ['1', 'i', 'l', '|'],
    'l': ['1', 'I', 'i', '|'],
    '5': ['S', 's'],
    'S': ['5', 's'],
    '8': ['B'],
    'B': ['8'],
    '2': ['Z'],
    'Z': ['2'],
    '6': ['G', 'b'],
    'G': ['6'],
    '-': ['_', '.', '—', '–'],
    '_': ['-', '.'],
}
