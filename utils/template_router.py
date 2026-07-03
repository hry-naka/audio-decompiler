import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE_DIR = PROJECT_ROOT / "templates"

# Weekday ordering map for boundary index evaluations
WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract ISO date pattern (YYYY-MM-DD) from given filename or file path.
    Example: ECN_2026-04-10-01.mp4 -> 2026-04-10
    """
    # Matches YYYY-MM-DD patterns
    date_pattern = r"(\d{4}-\d{2}-\d{2})"
    match = re.search(date_pattern, filename)
    
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d")
        except ValueError:
            return None
    return None


def match_range_template(weekday_str: str, program_dir: Path) -> Optional[Path]:
    """
    Scan directory for PTN2 range templates (e.g., 'Mon-Thu.html') 
    and evaluate if target weekday falls inside the boundary.
    """
    if not weekday_str in WEEKDAY_ORDER:
        return None
        
    current_idx = WEEKDAY_ORDER.index(weekday_str)
    
    # Iterate through all range-patternized templates
    for template_file in program_dir.glob("*-*.html"):
        try:
            # Splits "Mon-Thu" from "Mon-Thu.html"
            range_part = template_file.stem
            start_day, end_day = range_part.split("-")
            
            if start_day in WEEKDAY_ORDER and end_day in WEEKDAY_ORDER:
                start_idx = WEEKDAY_ORDER.index(start_day)
                end_idx = WEEKDAY_ORDER.index(end_day)
                
                # Check if current day is encapsulated in the range bounds
                if start_idx <= current_idx <= end_idx:
                    return template_file
        except ValueError:
            # Skip files that match regex but aren't structural valid ranges
            continue
            
    return None


def get_effective_template(program_name: str, file_identifier: str) -> Path:
    """
    Primary router logic: Resolves programmatic template path based on course rules.
    PTN1 (Exact match) takes absolute precedence over PTN2 (Range sequence).
    """
    program_name_upper = program_name.upper()
    program_dir = DEFAULT_TEMPLATE_DIR / program_name_upper
    
    # Extract date signature
    target_date = extract_date_from_filename(file_identifier)
    
    if not target_date:
        print(f"[*] Warning: Could not resolve valid date format in '{file_identifier}'. Falling back to default baseline.")
        return DEFAULT_TEMPLATE_DIR / "default_base.html"
        
    # Standardize to 3-char weekday identifier (e.g., 'Mon', 'Fri')
    weekday_str = target_date.strftime("%a")
    
    # Phase 1: Look for PTN1 (Exact day template match, e.g., 'Fri.html')
    exact_match = program_dir / f"{weekday_str}.html"
    if exact_match.exists():
        return exact_match
        
    # Phase 2: Look for PTN2 (Dynamic Range evaluation match, e.g., 'Mon-Thu.html')
    range_match = match_range_template(weekday_str, program_dir)
    if range_match and range_match.exists():
        return range_match
        
    # Phase 3: Global Fallback mechanism if no definitions are encountered
    fallback_path = DEFAULT_TEMPLATE_DIR / "default_base.html"
    if not fallback_path.parent.exists():
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
    if not fallback_path.exists():
        # Generate a minimal placeholder if missing
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html><html><body>{{ lesson_title }} (Fallback Base)</body></html>")
            
    return fallback_path


# ==========================================
# Self-Testing Sandbox Logic
# ==========================================
if __name__ == "__main__":
    print("[*] Running standalone diagnostic pipeline for template_router.py...")
    
    # Mock files mimicking real target paths
    test_cases = [
        ("ECN", "ECN_2026-04-06-13_40.mp4"),  # Mon -> Should map to Mon-Thu
        ("ECN", "ECN_2026-04-09-01.html"),     # Thu -> Should map to Mon-Thu
        ("ECN", "ECN_2026-04-10-01.mp4"),     # Fri -> Should map to Fri.html (PTN1)
        ("BBE", "BBE_2026-04-06.mp4")          # Non-existent directory example -> Fallback
    ]
    
    for prog, file_name in test_cases:
        resolved = get_effective_template(prog, file_name)
        print(f"  Input: [{prog}] {file_name:<26} -> Resolved Template: {resolved.name}")
