"""
Split PDF - Extract pages 133-136 from a PDF file
"""

from PyPDF2 import PdfReader, PdfWriter
import os

# Configuration
INPUT_PDF = "uploads/2024-12-03 NOBE II - Drawings.pdf"  # Change this to your PDF path
OUTPUT_PDF = "uploads/pages_133-136.pdf"
START_PAGE = 133  # First page to extract
END_PAGE = 136    # Last page to extract

def split_pdf():
    # Check if input file exists
    if not os.path.exists(INPUT_PDF):
        print(f"Error: File not found: {INPUT_PDF}")
        return
    
    # Read the PDF
    reader = PdfReader(INPUT_PDF)
    total_pages = len(reader.pages)
    
    print(f"Input PDF: {INPUT_PDF}")
    print(f"Total pages: {total_pages}")
    print(f"Extracting pages {START_PAGE} to {END_PAGE}...")
    
    # Validate page range
    if START_PAGE < 1 or END_PAGE > total_pages:
        print(f"Error: Invalid page range. PDF has {total_pages} pages.")
        return
    
    # Create a new PDF with selected pages
    writer = PdfWriter()
    
    # PyPDF2 uses 0-based indexing, so subtract 1 from page numbers
    for page_num in range(START_PAGE - 1, END_PAGE):
        writer.add_page(reader.pages[page_num])
        print(f"  Added page {page_num + 1}")
    
    # Save the output PDF
    with open(OUTPUT_PDF, "wb") as output_file:
        writer.write(output_file)
    
    print(f"\nDone! Saved to: {OUTPUT_PDF}")
    print(f"Extracted {END_PAGE - START_PAGE + 1} pages")

if __name__ == "__main__":
    split_pdf()

