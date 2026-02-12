"""
Auto-cleanup script for CAD Material Tracker
Deletes uploaded PDFs and generated images to free up disk space
"""

import os
import shutil
from pathlib import Path

# Folders to clean
CLEANUP_FOLDERS = [
    "uploads",           # Uploaded PDF files
    "output_images",     # Processed images
    "predicted_images",  # Roboflow annotated images
    "predicted_images_web",  # Web preview images
    "predicted_images2",  # Additional image folder
]

# Files/folders to KEEP (don't delete)
KEEP_FOLDERS = [
    "results",  # Keep CSV results
    "static",
    "templates",
    "venv",
    "venv312",
    "__pycache__",
]

def cleanup_files(keep_results=True):
    """
    Clean up uploaded PDFs and generated images
    
    Args:
        keep_results: If True, keeps the results/ folder with CSV files
    """
    total_deleted = 0
    total_size = 0
    
    print("🧹 Starting cleanup...")
    print("-" * 50)
    
    # Clean up folders
    for folder in CLEANUP_FOLDERS:
        folder_path = Path(folder)
        if folder_path.exists():
            # Calculate size before deletion
            folder_size = sum(f.stat().st_size for f in folder_path.rglob('*') if f.is_file())
            
            # Count files
            file_count = len(list(folder_path.rglob('*')))
            file_count = sum(1 for f in folder_path.rglob('*') if f.is_file())
            
            try:
                shutil.rmtree(folder_path)
                total_deleted += file_count
                total_size += folder_size
                print(f"✅ Deleted: {folder}/ ({file_count} files, {format_size(folder_size)})")
            except Exception as e:
                print(f"❌ Error deleting {folder}/: {e}")
        else:
            print(f"⏭️  Skipped: {folder}/ (doesn't exist)")
    
    # Recreate empty folders (so app doesn't crash)
    for folder in CLEANUP_FOLDERS:
        os.makedirs(folder, exist_ok=True)
    
    print("-" * 50)
    print(f"✨ Cleanup complete!")
    print(f"   Deleted: {total_deleted} files")
    print(f"   Freed: {format_size(total_size)}")
    
    if keep_results:
        print(f"\n💾 Results folder (CSV files) was kept intact")


def cleanup_results_only():
    """Clean up only the results CSV file (keep everything else)"""
    results_path = Path("results/results.csv")
    if results_path.exists():
        size = results_path.stat().st_size
        results_path.unlink()
        print(f"✅ Deleted: results/results.csv ({format_size(size)})")
    else:
        print("⏭️  No results.csv file found")


def format_size(size_bytes):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--results-only":
        cleanup_results_only()
    else:
        cleanup_files(keep_results=True)

