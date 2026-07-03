import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

import google.generativeai as genai
from utils.template_router import get_effective_template, extract_date_from_filename

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = PROJECT_ROOT / "prompts"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "daily"
TMP_DIR = PROJECT_ROOT / "tmp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


def configure_gemini() -> None:
    if not GEMINI_API_KEY:
        print("[Error] GEMINI_API_KEY missing in .env configurations.")
        sys.exit(1)
    genai.configure(api_key=GEMINI_API_KEY)


def prepare_audio_payload(input_path: Path) -> Path:
    """
    Smart Payload Router:
    - If input is a heavy/pseudo .mp4, losslessly extract the audio track to .m4a in 0.1s.
    - If input is already a native audio asset (.m4a, .mp3), bypass and stream directly.
    """
    ext = input_path.suffix.lower()
    
    if ext == ".mp4":
        output_path = TMP_DIR / f"extracted_{input_path.stem}.m4a"
        
        if output_path.exists():
            return output_path
            
        print(f"[*] Pseudo-video container (.mp4) detected. Extracting lossless audio stream to .m4a...")
        
        # -vn: Drop video track, -acodec copy: Demux/Remux the audio stream directly without re-encoding (Instant!)
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vn", "-acodec", "copy",
            str(output_path)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return output_path
        except subprocess.CalledProcessError:
            print("[Warning] Lossless extraction failed. Falling back to raw file stream deployment.")
            return input_path
    else:
        # Pass through for true audio files (.m4a, .mp3, etc.)
        return input_path


def read_prompt_card(program_name: str, template_name: str) -> str:
    if "Fri" in template_name:
        prompt_path = PROMPT_DIR / f"{program_name}_Fri.txt"
    else:
        prompt_path = PROMPT_DIR / f"{program_name}_Mon-Thu.txt"
        
    if not prompt_path.exists():
        prompt_path = PROMPT_DIR / f"{program_name}.txt"
        
    if not prompt_path.exists():
        print(f"[Error] Required prompt criteria profile missing for: {program_name}")
        sys.exit(1)
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def verify_and_clean_json(raw_response_text: str) -> str:
    cleaned = raw_response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def inject_data_to_template(template_path: Path, json_string_payload: str, meta: dict) -> str:
    with open(template_path, "r", encoding="utf-8") as tf:
        template_content = tf.read()
        
    template_content = template_content.replace("{{ program_id }}", str(meta.get("program_id", "")))
    template_content = template_content.replace("{{ broadcast_date }}", str(meta.get("broadcast_date", "")))
    
    injection_block = f'<script type="application/json" id="lesson-data">\n{json_string_payload}\n</script>'
    
    if "<!-- DATA_INJECTION_MARKER -->" in template_content:
        return template_content.replace("<!-- DATA_INJECTION_MARKER -->", injection_block)
    else:
        return template_content.replace("</body>", f"{injection_block}\n</body>")


def main():
    parser = argparse.ArgumentParser(description="Agnostic Audio Decoupler and Layout Compiler Core Pipeline")
    parser.add_argument("audio_path", type=str, help="Path to targeted input media file source")
    parser.add_argument("program_name", type=str, help="Target application template context profile key (e.g., ECN)")
    
    args = parser.parse_args()
    audio_path = Path(args.audio_path)
    program_name = args.program_name
    
    if not audio_path.exists():
        print(f"[Error] Specified sound asset path not found: {audio_path}")
        sys.exit(1)
        
    print(f"[+] Triggering pipeline for asset: {audio_path.name}")
    
    raw_broadcast_date = extract_date_from_filename(audio_path.name)
    broadcast_date_str = str(raw_broadcast_date).split(" ")[0].strip()
    
    chosen_template = get_effective_template(program_name, audio_path.name)
    print(f"[*] Routed Layout Destination Matrix ➔ {chosen_template.name}")
    
    meta_injections = {
        "program_id": program_name,
        "broadcast_date": broadcast_date_str
    }
    
    prompt_instruction = read_prompt_card(program_name, chosen_template.name)
    
    # Run the smart asset preparation layer (0.1s audio extraction if .mp4)
    target_payload_file = prepare_audio_payload(audio_path)
    
    configure_gemini()
    
    MODEL_NAME = "gemini-2.5-flash-lite"
    print(f"[*] Extracting structural telemetry using {MODEL_NAME}...")
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        
        print(f"[*] Uploading sound file to Google Media API pipeline...")
        uploaded_audio_file = genai.upload_file(path=target_payload_file)
        
        print("[*] Waiting for Google server side media processing to achieve ACTIVE state...")
        retries = 0
        while uploaded_audio_file.state.name == "PROCESSING":
            if retries >= 20:
                print("[Error] Media processing operation timed out on Google infrastructure side.")
                sys.exit(1)
            time.sleep(1.5)
            retries += 1
            uploaded_audio_file = genai.get_file(uploaded_audio_file.name)
            
        if uploaded_audio_file.state.name == "FAILED":
            print("[Error] Google Media API internal conversion failed for this asset source.")
            sys.exit(1)
            
        print(f"➔ [ACTIVE] Media index operation successful. Handing over to {MODEL_NAME}...")
        
        prompt_payload = [
            prompt_instruction, 
            uploaded_audio_file
        ]
        
        response = model.generate_content(prompt_payload)
        cleaned_json_str = verify_and_clean_json(response.text)
        
        final_html_output = inject_data_to_template(chosen_template, cleaned_json_str, meta_injections)
        
        output_filename = f"{program_name}_{broadcast_date_str}.html"
        save_path = OUTPUT_DIR / output_filename
        
        with open(save_path, "w", encoding="utf-8") as out_f:
            out_f.write(final_html_output)
            
        print(f"\n[💥SUCCESS] Generation lifecycle complete for {program_name} ({broadcast_date_str})!")
        print(f"➔ Static asset saved at: {save_path}")
        
    except Exception as e:
        print(f"\n[Error] Execution failure: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
