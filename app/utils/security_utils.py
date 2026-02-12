"""
Security helpers: path sanitization to prevent path traversal and unsafe file access.
"""

import os
from pathlib import Path


def safe_filename(filename: str) -> str | None:
    """
    Return a safe basename for saving to disk. Rejects path traversal and empty names.
    Returns None if the filename is unsafe.
    """
    if not filename or not filename.strip():
        return None
    # Use only the final component (no directories)
    base = os.path.basename(filename.strip())
    # Reject path traversal attempts
    if ".." in base or os.path.isabs(filename) or base != filename.strip():
        return None
    # Reject control chars and null
    if any(ord(c) < 32 for c in base):
        return None
    return base


def is_path_under_directory(file_path: str, directory: str) -> bool:
    """
    Return True if file_path is a path under directory (resolved, no traversal).
    """
    try:
        resolved_file = os.path.realpath(os.path.abspath(file_path))
        resolved_dir = os.path.realpath(os.path.abspath(directory))
        return resolved_file.startswith(resolved_dir + os.sep) or resolved_file == resolved_dir
    except (OSError, ValueError):
        return False


def validate_pdf_path_for_job(pdf_path: str, upload_folder: str) -> bool:
    """
    Return True if pdf_path exists, is a file, and is under upload_folder (no path traversal).
    """
    if not pdf_path or not isinstance(pdf_path, str):
        return False
    if not is_path_under_directory(pdf_path, upload_folder):
        return False
    return os.path.isfile(pdf_path)
