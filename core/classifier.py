import os
import tempfile
import shutil
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import joblib
import librosa

# Disable warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Paths relative to the core directory
CORE_DIR: Path = Path(__file__).resolve().parent
MODEL_PATH: Path = CORE_DIR / "classifier.joblib"
SCALER_PATH: Path = CORE_DIR / "scaler.joblib"

# Configuration (must match training parameters)
DURATION: int = 180       # First 3 minutes of audio
SR: int = 16000           # Sampling rate (16kHz)
N_MFCC: int = 20          # Number of MFCC features


def predict_genre(file_path_str: str) -> Optional[Dict[str, Any]]:
    """
    Predict whether the given audio file is "Music" or "Language".
    
    Args:
        file_path_str (str): Path to the target audio file.
        
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing prediction results and probabilities,
                                  or None if an error occurs.
    """
    file_path: Path = Path(file_path_str)
    if not file_path.exists():
        print(f"[Error] File not found: {file_path}")
        return None

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        print("[Error] Trained model or scaler file not found in the core directory.")
        return None

    # Load model and scaler
    model: Any = joblib.load(MODEL_PATH)
    scaler: Any = joblib.load(SCALER_PATH)

    # Use a temporary file for OneDrive compatibility (safe read-only processing)
    temp_dir: str = tempfile.gettempdir()
    temp_file_path: Path = Path(temp_dir) / f"temp_classify_{file_path.name}"

    try:
        shutil.copy2(file_path.resolve(), temp_file_path)

        # Load audio file (first 180 seconds)
        y, sr = librosa.load(str(temp_file_path), duration=DURATION, sr=SR)
        if len(y) == 0:
            print("[Warning] Loaded audio data is empty.")
            return None

        # Extract MFCC features
        mfcc: np.ndarray = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean: np.ndarray = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # Scale features
        mfcc_scaled: np.ndarray = scaler.transform(mfcc_mean)

        # Predict probability and class
        probabilities: np.ndarray = model.predict_proba(mfcc_scaled)[0]
        pred_class: int = model.predict(mfcc_scaled)[0]

        return {
            "file_name": file_path.name,
            "prediction": "Music" if pred_class == 0 else "Language",
            "prob_music": float(probabilities[0] * 100),
            "prob_language": float(probabilities[1] * 100)
        }

    except Exception as e:
        print(f"[Error] Analysis failed: {e}")
        return None

    finally:
        # Cleanup temporary file
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
