"""
Ground Truth vs App Pipeline Comparison
========================================
Edit the CONFIG below, then run:   python ground_truth_counter.py
"""

import os, sys, re, csv, gc, time, json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ╔══════════════════════════════════════════════════════════════╗
# ║  EDIT THIS CONFIG
# ╚══════════════════════════════════════════════════════════════╝

PDF_PATH = r"D:\absolute-builders\10R_INTERIOR.pdf"

# Tags to search for (add as many as you want)
TAGS_TO_TEST = [
    "A_WB-3",
    "A_WD-1",
    "A_WD-5",
]

# Page range (set both to same number to test a single page, or None for all pages)
START_PAGE = 1   # e.g. 36  (1-indexed, None = first page)
END_PAGE = 96     # e.g. 36  (1-indexed, None = last page)

# Set to True to also run the Roboflow app pipeline for comparison
RUN_APP_PIPELINE = True

# Output CSV file
OUTPUT_CSV = "ground_truth_comparison.csv"

# ════════════════════════════════════════════════════════════════

import fitz  # PyMuPDF
from app.core.config import ALWAYS_PYMUPDF
from app.services.text_extractor import (
    image_coords_to_pdf_coords, extract_text_from_pdf_region,
    is_pymupdf_available, get_fitz
)
from app.services.pdf_processor import (
    render_page_to_image, detect_drawings_on_image, process_ocr_and_tags,
    clean_tag_text, find_all_tag_positions_cleaned, positions_are_same,
    is_point_inside_bbox, distance_to_region_center,
)
from app.services.ai_service import is_roboflow_available


def normalize_tag(tag):
    tag = tag.strip().upper()
    for ch in '\u2010\u2011\u2012\u2013\u2014\u00ad':
        tag = tag.replace(ch, '-')
    return tag


# ═══════════════════════════════════════════════════════════
# GROUND TRUTH — PyMuPDF search_for (deterministic, full page)
# ═══════════════════════════════════════════════════════════

def ground_truth_count(pdf_path, tags, start_page=None, end_page=None):
    doc = fitz.open(pdf_path)
    
    # Structure: results[tag][page_num] = count
    results = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    s = (start_page or 1) - 1
    e = min(end_page or len(doc), len(doc))

    for i in range(s, e):
        page = doc[i]
        page_num = i + 1
        
        for tag in tags:
            search_tag = normalize_tag(tag)
            hits = page.search_for(search_tag, quads=False)
            count = len(hits)
            if count > 0:
                results[tag][page_num] = count
                totals[tag] += count

    doc.close()
    return results, totals


# ═══════════════════════════════════════════════════════════
# APP PIPELINE — Roboflow → Text Extraction → Tag Match → Dedup
# ═══════════════════════════════════════════════════════════

def app_pipeline_count(pdf_path, tags, start_page=None, end_page=None):
    if not is_roboflow_available():
        print("  ⚠ Roboflow not available — skipping app pipeline")
        return {}, {} if tags else (None, 0)

    fitz_mod = get_fitz()
    doc = fitz_mod.open(pdf_path)
    total_pages = len(doc)
    s = start_page or 1
    e = min(end_page or total_pages, total_pages)

    all_regions_data = []

    print(f"  Scanning pages {s} to {e}...")

    for page_num in range(s, e + 1):
        page = doc[page_num - 1]
        print(f"  Page {page_num}/{e}...", end="", flush=True)
        t0 = time.time()

        # 1. Render
        try:
            img_original, resolution = render_page_to_image(page, page_num, use_pymupdf=True)
        except Exception as e:
            print(f" render error: {e}")
            continue

        if not img_original or not resolution:
            print(" no image")
            continue

        # 2. Detect
        try:
            _, cropped_drawings, _ = detect_drawings_on_image(img_original, page_num, resolution)
        except Exception as e:
            print(f" detection error: {e}")
            continue

        n = len(cropped_drawings) if cropped_drawings else 0
        if not cropped_drawings:
            print(f" 0 drawings ({time.time()-t0:.1f}s)")
            del img_original
            gc.collect()
            continue

        page_rect = page.rect
        pdf_size = (page_rect.width, page_rect.height)
        image_size = img_original.size

        # 3. Extract Text from Regions
        for i, crop_data in enumerate(cropped_drawings):
            pdf_bbox = image_coords_to_pdf_coords(crop_data["bbox"], image_size, pdf_size, resolution)
            try:
                text, word_positions, mupdf_rect = extract_text_from_pdf_region(
                    pdf_path, page_num, pdf_bbox, pdf_size, return_positions=True
                )
            except Exception:
                text, word_positions, mupdf_rect = "", [], None

            all_regions_data.append({
                'drawing_index': i + 1,
                'word_positions': word_positions,
                'mupdf_rect': mupdf_rect,
                'page': page_num,
                'confidence': crop_data["confidence"],
                'bbox': crop_data["bbox"]
            })

        print(f" {n} drawings ({time.time()-t0:.1f}s)")
        del img_original, cropped_drawings
        gc.collect()

    doc.close()

    # 4. Match Tags & Dedup
    print(f"\n  Processing detections for {len(tags)} tags...")
    all_detections = []
    
    # Find all occurrences of all tags in all regions
    for rd in all_regions_data:
        if not rd['word_positions']:
            continue
        
        for tag in tags:
            # Note: finding relies on find_all_tag_positions_cleaned being updated in app code
            positions = find_all_tag_positions_cleaned(tag, rd['word_positions'])
            for pos in positions:
                all_detections.append({
                    'result': {'tag': tag, 'page': rd['page'], 'drawing_index': rd['drawing_index']},
                    'position': pos,
                    'mupdf_rect': rd['mupdf_rect'],
                    'drawing_index': rd['drawing_index'],
                })

    # Group detections BY TAG first (Critical fix for multi-tag interference)
    detections_by_tag = defaultdict(list)
    for det in all_detections:
        detections_by_tag[det['result']['tag']].append(det)

    final_results = []
    for tag, tag_detections in detections_by_tag.items():
        # Dedup positions within this tag only
        unique_positions = []
        for det in tag_detections:
            pos = det['position']
            matched = False
            for u in unique_positions:
                # Use tolerance=5 (matches strict dedup fix)
                if positions_are_same(pos, u['position'], tolerance=5):
                    u['detections'].append(det)
                    matched = True
                    break
            if not matched:
                unique_positions.append({'position': pos, 'detections': [det]})
        
        # Resolve multiples at same position
        for u in unique_positions:
            tag_pos = u['position']
            dets = u['detections']
            if len(dets) == 1:
                final_results.append(dets[0]['result'])
            else:
                containing = [d for d in dets if d['mupdf_rect'] and is_point_inside_bbox(tag_pos, d['mupdf_rect'])]
                if len(containing) <= 1:
                    final_results.append((containing or dets)[0]['result'])
                else:
                    best = min(containing, key=lambda d: distance_to_region_center(tag_pos, d['mupdf_rect']))
                    final_results.append(best['result'])

    # Compile results
    results = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for r in final_results:
        tag = r['tag']
        page = r['page']
        results[tag][page] += 1
        totals[tag] += 1

    return results, totals


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*60}")
    print(f"  GROUND TRUTH COMPARISON (Multi-Tag)")
    print(f"  PDF: {PDF_PATH}")
    print(f"  TAGS: {', '.join(TAGS_TO_TEST)}")
    print(f"{'═'*60}")

    # Ground truth
    print(f"\n▶ GROUND TRUTH (PyMuPDF search — deterministic)")
    gt_results, gt_totals = ground_truth_count(PDF_PATH, TAGS_TO_TEST, START_PAGE, END_PAGE)
    for tag in TAGS_TO_TEST:
        print(f"  {tag:<10}: {gt_totals[tag]}")

    # App pipeline
    app_results, app_totals = {}, {}
    if RUN_APP_PIPELINE:
        print(f"\n▶ APP PIPELINE (Roboflow → Text → Match → Dedup)")
        app_results, app_totals = app_pipeline_count(PDF_PATH, TAGS_TO_TEST, START_PAGE, END_PAGE)
        print("  Totals (App):")
        for tag in TAGS_TO_TEST:
            print(f"  {tag:<10}: {app_totals.get(tag, 0)}")

    # Comparison Output
    print(f"\n{'═'*60}")
    print(f"  DETAILED COMPARISON")
    print(f"{'═'*60}")

    # Collect all pages that have any hits
    all_pages = set()
    for tag in TAGS_TO_TEST:
        all_pages.update(gt_results[tag].keys())
        if RUN_APP_PIPELINE:
            all_pages.update(app_results.get(tag, {}).keys())
    
    sorted_pages = sorted(all_pages)

    if not sorted_pages:
        print("  No tags found on any processed pages.")
    else:
        print(f"  {'TAG':<12} {'PAGE':<8} {'APP':>8} {'TRUTH':>8} {'DIFF':>8}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for tag in TAGS_TO_TEST:
            tag_pages = sorted(set(gt_results[tag].keys()) | set(app_results.get(tag, {}).keys()))
            
            for p in tag_pages:
                a = app_results.get(tag, {}).get(p, 0) if RUN_APP_PIPELINE else 0
                g = gt_results[tag].get(p, 0)
                diff = a - g
                
                # Only show rows with discrepancies or relevant data
                mark = " ✓" if diff == 0 else ""
                print(f"  {tag:<12} {p:<8} {a:>8} {g:>8} {diff:>+8}{mark}")
            
            # Tag Total
            a_tot = app_totals.get(tag, 0) if RUN_APP_PIPELINE else 0
            g_tot = gt_totals[tag]
            print(f"  {tag+' ALL':<12} {'TOTAL':<8} {a_tot:>8} {g_tot:>8} {a_tot - g_tot:>+8}\n")

    # CSV Output
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["Tag", "Page", "App Count", "Ground Truth", "Difference"] if RUN_APP_PIPELINE else ["Tag", "Page", "Ground Truth"]
        w.writerow(header)
        
        for tag in TAGS_TO_TEST:
            tag_pages = sorted(set(gt_results[tag].keys()) | set(app_results.get(tag, {}).keys()))
            for p in tag_pages:
                g = gt_results[tag].get(p, 0)
                if RUN_APP_PIPELINE:
                    a = app_results.get(tag, {}).get(p, 0)
                    w.writerow([tag, p, a, g, a - g])
                else:
                    w.writerow([tag, p, g])
            
            # Total row per tag
            g_tot = gt_totals[tag]
            if RUN_APP_PIPELINE:
                a_tot = app_totals.get(tag, 0)
                w.writerow([tag + " (TOTAL)", "ALL", a_tot, g_tot, a_tot - g_tot])
            else:
                w.writerow([tag + " (TOTAL)", "ALL", g_tot])

    print(f"  CSV saved: {OUTPUT_CSV}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
