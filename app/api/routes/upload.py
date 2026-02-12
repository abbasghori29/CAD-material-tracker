"""
Upload routes - Handle file uploads (PDF and tags)
"""

import os
import shutil
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import JSONResponse
import pandas as pd

from app.core.config import UPLOAD_FOLDER, TAG_DESCRIPTIONS
from app.api.deps.auth import get_current_user
from app.utils.security_utils import safe_filename
from app.services.text_extractor import is_pymupdf_available, get_fitz
import pdfplumber

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.post("/upload-tags")
async def upload_tags(file: UploadFile = File(...)):
    """Upload CSV or XLSX file with tag list"""
    global TAG_DESCRIPTIONS
    
    try:
        # Check file extension
        filename = file.filename.lower()
        if not (filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.xls')):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "File must be CSV, XLSX, or XLS format"}
            )
        
        try:
            # Read file content into memory
            content = await file.read()
            file_size = len(content)
            print(f"Received file: {file.filename}, Size: {file_size} bytes")
            
            if file_size < 10:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "File is too small or empty"}
                )

            file_buffer = BytesIO(content)

            if filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file_buffer, engine='python', sep=None, on_bad_lines='skip')
                except Exception as e:
                    print(f"CSV read error: {e}, trying default")
                    file_buffer.seek(0)
                    df = pd.read_csv(file_buffer)
                
                cols_upper = [str(c).upper() for c in df.columns]
                
                if 'TAGS' not in cols_upper:
                     if len(df.columns) >= 2:
                         file_buffer.seek(0)
                         df = pd.read_csv(file_buffer, header=None, names=['Tags', 'Material Type'])
                     else:
                         df.rename(columns={df.columns[0]: 'Tags'}, inplace=True)
                else:
                    df.columns = [str(c).strip().title() for c in df.columns]
                    
            else:  # .xlsx or .xls
                try:
                    if filename.endswith('.xlsx'):
                        df = pd.read_excel(file_buffer, engine='openpyxl')
                    else:
                        try:
                            df = pd.read_excel(file_buffer, engine='openpyxl')
                        except:
                            file_buffer.seek(0)
                            df = pd.read_excel(file_buffer)
                except ImportError:
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "error": "Excel support not installed. Please install openpyxl: pip install openpyxl"}
                    )
                except Exception as e:
                    try:
                        file_buffer.seek(0)
                        df = pd.read_excel(file_buffer)
                    except Exception as e2:
                        raise Exception(f"Failed to read Excel file: {str(e2)}")
                
                cols_upper = [str(c).upper() for c in df.columns]
                
                if 'TAGS' not in cols_upper:
                    if len(df.columns) >= 2:
                        file_buffer.seek(0)
                        try:
                            if filename.endswith('.xlsx'):
                                df = pd.read_excel(file_buffer, engine='openpyxl', header=None, names=['Tags', 'Material Type'])
                            else:
                                df = pd.read_excel(file_buffer, header=None, names=['Tags', 'Material Type'])
                        except:
                            file_buffer.seek(0)
                            df = pd.read_excel(file_buffer, header=None, names=['Tags', 'Material Type'])
                    else:
                        df.rename(columns={df.columns[0]: 'Tags'}, inplace=True)
                else:
                    df.columns = [str(c).strip().title() for c in df.columns]
                
            # Clean data
            df['Tags'] = df['Tags'].astype(str).str.strip().str.upper()
            if 'Material Type' in df.columns:
                df['Material Type'] = df['Material Type'].fillna('Unknown').astype(str).str.strip()
            else:
                df['Material Type'] = 'Unknown'
            
            # Update global TAG_DESCRIPTIONS
            from app.core import config
            config.TAG_DESCRIPTIONS = pd.Series(df['Material Type'].values, index=df['Tags']).to_dict()
            
            # Remove invalid keys
            config.TAG_DESCRIPTIONS = {k: v for k, v in config.TAG_DESCRIPTIONS.items() if k and k.lower() != 'nan'}
            
            print(f"Loaded {len(config.TAG_DESCRIPTIONS)} tags: {list(config.TAG_DESCRIPTIONS.keys())[:5]}...")
            return {
                "success": True,
                "message": f"Loaded {len(config.TAG_DESCRIPTIONS)} tags successfully",
                "tags": list(config.TAG_DESCRIPTIONS.keys()),
                "count": len(config.TAG_DESCRIPTIONS)
            }
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error parsing file: {error_trace}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Error parsing file content: {str(e)}"}
            )
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error processing file: {error_trace}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error processing file: {str(e)}"}
        )


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF file. Filename is sanitized to prevent path traversal."""
    raw_name = file.filename or ""
    if not raw_name.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "File must be a PDF"}
        )
    safe_name = safe_filename(raw_name)
    if not safe_name:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Invalid filename"}
        )
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Use PyMuPDF for faster page count
    if is_pymupdf_available():
        fitz = get_fitz()
        pdf_doc = fitz.open(file_path)
        page_count = len(pdf_doc)
        pdf_doc.close()
    else:
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
    
    return {"filename": safe_name, "path": file_path, "pages": page_count}
