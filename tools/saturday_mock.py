import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Ensure environmental loaders are handled safely
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

import google.generativeai as genai

# ==========================================
# Configuration & Paths
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "outputs" / "daily"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "saturday"
PROMPT_DIR = PROJECT_ROOT / "prompts"

# Auto-create necessary directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_gemini() -> None:
    """Initialize Gemini API."""
    if not GEMINI_API_KEY:
        print("\n[Error] GEMINI_API_KEY is not configured in the .env file.")
        sys.exit(1)
    genai.configure(api_key=GEMINI_API_KEY)


def get_weekly_html_files(program_name: str, days_back: int = 6) -> List[Path]:
    """Scan and collect daily HTML outputs generated within the past week for a specific program."""
    collected_files = []
    today = datetime.now()
    
    if not INPUT_DIR.exists():
        return []
        
    for html_file in INPUT_DIR.glob(f"{program_name}_*.html"):
        # Extract date string from filename (e.g., ECN_2026-04-01-13.html -> 2026-04-01-13)
        # Fallback to file modification time if name pattern varies
        try:
            file_mtime = datetime.fromtimestamp(html_file.stat().st_mtime)
            if today - file_mtime <= timedelta(days=days_back):
                collected_files.append(html_file)
        except Exception:
            pass
            
    # Sort files chronically based on creation/modification time
    collected_files.sort(key=lambda x: x.stat().st_mtime)
    return collected_files


def extract_anki_json_from_html(html_path: Path) -> List[Dict[str, str]]:
    """Extract the embedded hidden Anki JSON script block from a generated daily HTML file."""
    cards = []
    if not html_path.exists():
        return cards
        
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Use Regex to extract the raw json block inside <script id="anki-data">
    pattern = r'<script\s+type="application/json"\s+id="anki-data">(.*?)</script>'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        try:
            json_data = json.loads(match.group(1).strip())
            if "cards" in json_data:
                cards.extend(json_data["cards"])
        except json.JSONDecodeError:
            print(f"[Warning] Failed to parse embedded Anki JSON in: {html_path.name}")
            
    return cards


def clean_html_output(raw_text: str) -> str:
    """Strip out markdown code block wrappers if included in the response."""
    text = raw_text.strip()
    if text.startswith("```html"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def generate_saturday_mock_prompt(program_name: str, weekly_cards: List[Dict[str, str]]) -> str:
    """Assemble a rigorous prompt demanding Gemini to synthesis a comprehensive weekend review exam."""
    cards_summary_str = json.dumps(weekly_cards, ensure_ascii=False, indent=2)
    
    # Read common base rules if available to preserve dark mode / visual alignment
    common_path = PROMPT_DIR / "common.txt"
    common_rules = ""
    if common_path.exists():
        with open(common_path, "r", encoding="utf-8") as f:
            common_rules = f.read()

    prompt = f"""{common_rules}

=========================================
[SATURDAY COMPREHENSIVE WEEKEND EXAM GENERATION]
=========================================
You are acting as the Chief Examiner for the {program_name} course. 
Your goal is to build a comprehensive, challenging 30-minute synthesis test based on the concepts learned throughout this week.

Below is the JSON summary of core vocabularies and phrases that the user encountered from Monday to Friday:
{cards_summary_str}

### 🧠 ROADMAP FOR THE SATURDAY TEST (30-Minute Volume)
Generate a beautiful, responsive dark-mode HTML containing the following distinct examination evaluation blocks:

1. **<header> / Weekend Evaluation Sheet**:
   - Title: "Weekly Synthesis Exam: {program_name} (Weekend Session)"
   - Brief motivational instruction note emphasizing application and memory retention.

2. **<section> / Part 1: Advanced Composition & Grammar Synthesis (応用表現・総合問題)**:
   - Do NOT just copy-paste the daily cards. Combine 2 or 3 of this week's learned grammar formulas or terms into brand new, complex sentences or short paragraph translations.
   - Provide 3 long-form challenge translation problems. Include an interactive HTML '<details><summary>Click to view model answer</summary>...</details>' toggle wrapper containing the correct answer and grammatical breakdown for each.

3. **<section> / Part 2: Listening & Context Simulation (シチュエーション読解・聴き取りシミュレーション)**:
   - Create a fictional business or casual dialogue scenario (depending on whether it is English/Chinese) utilizing at least 5 major keywords listed in the weekly cards data above.
   - Follow up with 2 reading comprehension multiple-choice questions testing situational nuances.

4. **<section> / Part 3: Weakness Drill (弱点克服集中ノック)**:
   - Design a rapid-fire quiz section for quick syntax or vocabulary retention.

### 🚫 OUTPUT RESTRICTIONS
- Return ONLY valid raw HTML. No markdown code blocks, no chat preambles. Start with '<!DOCTYPE html>'.
"""
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan weekly daily outputs and compile a dedicated weekend synthesis exam file (Phase 3)."
    )
    parser.add_argument(
        "program_name",
        type=str,
        help="Target prefix identifier of the course (e.g., ECN, BBE)"
    )
    
    args = parser.parse_args()
    program_name = args.program_name.upper()
    
    print(f"\n[+] Initiating Saturday compilation cycle for program: {program_name}")
    
    # 1. Gather all files modified within the past week
    weekly_files = get_weekly_html_files(program_name)
    if not weekly_files:
        print(f"[Warning] No daily outputs found for {program_name} within the past week track. Check your outputs/daily/ folder.")
        # For testing purposes, we can loosen the filter to collect all historical logs if empty
        print("[*] Re-scanning folder ignoring time constraints for debug/testing purpose...")
        weekly_files = list(INPUT_DIR.glob(f"{program_name}_*.html"))
        
    if not weekly_files:
        print("[Error] No input files found to extract data from. Execution terminated.")
        sys.exit(1)
        
    print(f"[*] Found {len(weekly_files)} daily HTML assets to compile.")
    for f in weekly_files:
        print(f"   ➔ {f.name}")
        
    # 2. Extract JSON payloads from HTML metadata
    all_weekly_cards = []
    for f in weekly_files:
        cards = extract_anki_json_from_html(f)
        all_weekly_cards.extend(cards)
        
    print(f"[*] Extracted a total of {len(all_weekly_cards)} unique reference terms from metadata files.")
    
    if not all_weekly_cards:
        print("[Warning] Embedded script tags contain no card blocks. Synthesizing test based on general course parameters.")
        
    # 3. Assemble and dispatch prompt to Gemini
    configure_gemini()
    prompt = generate_saturday_mock_prompt(program_name, all_weekly_cards)
    
    date_str = time.strftime("%Y%m%d")
    output_path = OUTPUT_DIR / f"SAT_{program_name}_{date_str}.html"
    
    try:
        print(f"[*] Dispatching synthesis request to Gemini (gemini-2.5-flash)...")
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        response = model.generate_content(prompt)
        
        if response.text:
            html_content = clean_html_output(response.text)
            
            with open(output_path, "w", encoding="utf-8") as out_f:
                out_f.write(html_content)
                
            print(f"\n[💥SUCCESS] Saturday synthesis weekend test generated successfully!")
            print(f"➔ Saved at: {output_path}")
        else:
            print("[Error] Gemini API did not yield a valid output response.")
    except Exception as e:
        print(f"[Error] Failed during generation sequence: {e}")


if __name__ == "__main__":
    main()
