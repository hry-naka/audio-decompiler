import os
import sys
import hmac
import hashlib
import time
import base64
import warnings
import tempfile
import shutil
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
import requests

# Disable warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables from .env (located in project root)
from dotenv import load_dotenv
env_path: Path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import librosa
import soundfile as sf

# ==========================================
# Configuration with Strict Auto-Stripping
# ==========================================
def get_clean_env(key: str, default: str = "") -> str:
    """Load env variable and strictly strip whitespace and any surrounding quotes"""
    val = os.getenv(key, default)
    if val:
        val = val.strip().strip("'").strip('"')
    return val

ACR_HOST: str = get_clean_env("ACR_HOST", "identify-ap-southeast-1.acrcloud.com")
ACR_ACCESS_KEY: str = get_clean_env("ACR_ACCESS_KEY")
ACR_ACCESS_SECRET: str = get_clean_env("ACR_ACCESS_SECRET")

# Set paths relative to the project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Audio recognition settings
DEFAULT_SAMPLE_INTERVAL: int = 180  # Interval to extract audio clips in seconds (3 mins)
DEFAULT_CLIP_DURATION: int = 12     # Clip length to send to API in seconds
SR: int = 8000                      # Sampling rate (8kHz is sufficient for music recognition)


def create_acr_signature(data_to_sign: str, secret: str) -> str:
    """Generate HMAC-SHA1 signature for ACRCloud API"""
    key: bytes = secret.encode("utf-8")
    msg: bytes = data_to_sign.encode("utf-8")
    return base64.b64encode(
        hmac.new(key, msg, digestmod=hashlib.sha1).digest()
    ).decode("utf-8")


def recognize_audio_clip(clip_path: str, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Send a short audio clip to ACRCloud API to identify the song"""
    if not ACR_ACCESS_KEY or not ACR_ACCESS_SECRET:
        print(
            "\n[Error] ACR_ACCESS_KEY or ACR_ACCESS_SECRET is not configured in .env file."
        )
        sys.exit(1)

    url: str = f"https://{ACR_HOST}/v1/identify"

    timestamp: str = str(int(time.time()))
    signature_version: str = "1"
    string_to_sign: str = f"POST\n/v1/identify\n{ACR_ACCESS_KEY}\naudio\n{signature_version}\n{timestamp}"
    signature: str = create_acr_signature(string_to_sign, ACR_ACCESS_SECRET)

    clip_file_path = Path(clip_path)
    if not clip_file_path.exists():
        return None

    # CRITICAL FIX: The parameter name for the audio file MUST be "sample", not "audio"
    files: Dict[str, Any] = {
        "sample": (clip_file_path.name, open(clip_path, "rb"), "audio/wav")
    }
    
    # Cast all form-data values to strings for robust multipart transmission
    data: Dict[str, str] = {
        "access_key": ACR_ACCESS_KEY,
        "sample_bytes": str(os.path.getsize(clip_path)),
        "timestamp": timestamp,
        "signature": signature,
        "data_type": "audio",
        "signature_version": signature_version,
    }

    try:
        response: requests.Response = requests.post(url, files=files, data=data, timeout=15)
        response.encoding = "utf-8"
        
        if response.status_code == 200:
            res_json = response.json()
            if verbose:
                status = res_json.get("status", {})
                print(f"\n[Debug] ACRCloud Code: {status.get('code')} | Msg: {status.get('msg')}")
            return res_json
        else:
            print(f"\n[Warning] HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"\n[Warning] ACRCloud API request failed: {e}")
    return None


def parse_acr_result(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Parse JSON metadata returned from ACRCloud API"""
    if not result_json or result_json.get("status", {}).get("code") != 0:
        return None  # Identification failed

    metadata: Dict[str, Any] = result_json.get("metadata", {})
    music_list: List[Dict[str, Any]] = metadata.get("music", [])
    if not music_list:
        return None

    music: Dict[str, Any] = music_list[0]  # Take the result with the highest confidence score
    title: str = music.get("title", "Unknown Title")
    artists: str = ", ".join(
        [artist.get("name") for artist in music.get("artists", [])]
    )
    album: str = music.get("album", {}).get("name", "Unknown Album")

    # Fetch external streaming link (Spotify) if available
    external_metadata: Dict[str, Any] = music.get("external_metadata", {})
    spotify_id: Optional[str] = external_metadata.get("spotify", {}).get("track", {}).get("id")
    spotify_url: Optional[str] = (
        f"https://open.spotify.com/track/{spotify_id}" if spotify_id else None
    )

    return {
        "title": title,
        "artists": artists,
        "album": album,
        "spotify_url": spotify_url,
    }


def process_program(file_path_str: str, sample_interval: int, clip_duration: int, debug: bool) -> None:
    """Scan the audio file (mp4) and generate a playlist"""
    file_path: Path = Path(file_path_str)
    print(f"[*] Starting playlist extraction for: {file_path.name}")
    print(f"[*] Configuration: interval={sample_interval}s, clip_duration={clip_duration}s, debug={debug}")

    # Parse program name and date from filename
    parts: List[str] = file_path.stem.split("_")
    program_name: str = parts[0]
    date_str: str = parts[1] if len(parts) > 1 else time.strftime("%Y-%m-%d")
    date_str = "-".join(date_str.split("-")[:3])

    output_file_name: str = f"{program_name}_{date_str}.md"
    output_path: Path = OUTPUT_DIR / output_file_name

    # Copy to temporary folder for safe read-only processing
    temp_dir: str = tempfile.gettempdir()
    temp_file_path: Path = Path(temp_dir) / f"temp_playlist_{file_path.name}"

    try:
        shutil.copy2(file_path.resolve(), temp_file_path)

        # Get total duration of the audio
        duration: float = librosa.get_duration(path=str(temp_file_path))
        print(f"[*] Total duration: {duration/60:.1f} minutes")

        playlist: List[Dict[str, Any]] = []
        seen_songs: Set[str] = set()  # Prevent duplicate tracks

        # Scan every sample_interval, starting 10 seconds into the file
        current_time: float = 10.0
        while current_time < duration:
            # Inline progress logging
            progress_msg: str = f"\r[~] Analyzing timestamp: {int(current_time/60)}m / {int(duration/60)}m..."
            sys.stdout.write(progress_msg)
            sys.stdout.flush()

            # Load short clip
            y, sr = librosa.load(
                str(temp_file_path),
                offset=current_time,
                duration=clip_duration,
                sr=SR,
            )

            # Export clip as temporary WAV file (Mono, PCM_16)
            temp_clip_path: Path = Path(temp_dir) / "temp_clip.wav"
            sf.write(temp_clip_path, y, sr, format="WAV", subtype="PCM_16")

            # Call API and parse response
            raw_result: Optional[Dict[str, Any]] = recognize_audio_clip(str(temp_clip_path), verbose=debug)
            song_info: Optional[Dict[str, Any]] = parse_acr_result(raw_result)

            if song_info:
                song_key: str = f"{song_info['title']}_{song_info['artists']}"
                if song_key not in seen_songs:
                    seen_songs.add(song_key)

                    # Calculate approximate timestamp
                    min_val: int = int(current_time // 60)
                    sec_val: int = int(current_time % 60)
                    song_info["timestamp"] = f"{min_val:02d}:{sec_val:02d}"

                    playlist.append(song_info)
                    print(
                        f"\n[+] Detected: [{song_info['timestamp']}] {song_info['artists']} - {song_info['title']}"
                    )

            # Clean up the clip
            if temp_clip_path.exists():
                os.remove(temp_clip_path)

            current_time += sample_interval

        print("\n[*] Exporting playlist to markdown...")
        save_playlist_to_markdown(program_name, date_str, playlist, output_path)

    except Exception as e:
        print(f"\n[Error] Playlist extraction failed: {e}")
    finally:
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


def save_playlist_to_markdown(program: str, date: str, playlist: List[Dict[str, Any]], output_path: Path) -> None:
    """Save playlist in a clean, consistent markdown format"""
    markdown_content: str = f"""# {program} - Playlist

- **Broadcast Date**: {date}
- **Generated At**: {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Total Tracks**: {len(playlist)}

---

## Tracklist

"""
    if not playlist:
        markdown_content += "No tracks detected in this broadcast.\n"
    else:
        for idx, song in enumerate(playlist, 1):
            spotify_link: str = (
                f"[Spotify Link]({song['spotify_url']})"
                if song["spotify_url"]
                else "-"
            )
            markdown_content += f"""### {idx:02d}. {song['title']}
- **Artist**: {song['artists']}
- **Album**: *{song['album']}*
- **Estimated Airtime**: `~{song['timestamp']}`
- **Streaming**: {spotify_link}

"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"[+] Playlist successfully saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract music playlist from a radio broadcast audio file using ACRCloud API."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the recorded audio file (mp4/m4a/etc.)."
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=DEFAULT_SAMPLE_INTERVAL,
        help=f"Sampling interval to check songs in seconds (default: {DEFAULT_SAMPLE_INTERVAL})."
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=DEFAULT_CLIP_DURATION,
        help=f"Duration of each audio clip sent to API in seconds (default: {DEFAULT_CLIP_DURATION})."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging to print API responses from ACRCloud."
    )
    
    args: argparse.Namespace = parser.parse_args()
    
    process_program(
        file_path_str=args.file_path,
        sample_interval=args.interval,
        clip_duration=args.duration,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
