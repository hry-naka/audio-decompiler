import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure project root is in path for importing core modules
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.classifier import predict_genre


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a radio broadcast audio file into 'Music' or 'Language' using a trained SVM model."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the audio file (mp4/m4a/etc.) to be classified."
    )
    
    args: argparse.Namespace = parser.parse_args()
    
    print(f"[*] Classifying file: {Path(args.file_path).name}")
    result: Optional[Dict[str, Any]] = predict_genre(args.file_path)
    
    if result:
        print("\n" + "=" * 42)
        print(f"🎯 Prediction: [ {result['prediction']} ]")
        print("=" * 42)
        print(f" ── Music Probability    : {result['prob_music']:.2f} %")
        print(f" ── Language Probability : {result['prob_language']:.2f} %")
        print("=" * 42 + "\n")


if __name__ == "__main__":
    main()