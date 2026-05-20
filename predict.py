import sys
import os
import tempfile
import shutil
import warnings
from pathlib import Path

# librosaのFutureWarningやUserWarningを非表示にする（インポート前に設定）
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import librosa
import numpy as np
import joblib

# ==========================================
# 設定（学習時と完全に一致させる必要があります）
# ==========================================
MODEL_PATH = Path("./classifier.joblib")
SCALER_PATH = Path("./scaler.joblib")

DURATION = 180       # 冒頭3分（180秒）を対象にする
SR = 16000           # サンプリングレート 16kHz
N_MFCC = 20          # 特徴量の次元数

def predict_single_file(file_path):
    """
    指定された1つのmp4ファイルを読み込み、音楽か語学かを判定する。
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ ファイルが見つかりません: {file_path}")
        return None

    # モデルとスケールの読み込み
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        print("❌ 学習済みのモデル、またはスケールファイルが見つかりません。先に train.py を実行してください。")
        return None

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # OneDrive対策：ローカルの一時フォルダにコピーして安全に解析
    temp_dir = tempfile.gettempdir()
    temp_file_path = Path(temp_dir) / f"temp_predict_{file_path.name}"

    try:
        real_src_path = file_path.resolve()
        shutil.copy2(real_src_path, temp_file_path)

        # 音声のロード
        y, sr = librosa.load(str(temp_file_path), duration=DURATION, sr=SR)
        if len(y) == 0:
            print("⚠️ 音声データが空です。")
            return None

        # 特徴量の抽出（MFCC）
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # データの標準化
        mfcc_scaled = scaler.transform(mfcc_mean)

        # 確率予測 (probability=True に設定した恩恵がここで受けられます)
        probabilities = model.predict_proba(mfcc_scaled)[0]
        
        # 予測クラスの決定 (0: Music, 1: Language)
        pred_class = model.predict(mfcc_scaled)[0]

        result = {
            "file_name": file_path.name,
            "prediction": "Music" if pred_class == 0 else "Language",
            "prob_music": probabilities[0] * 100,
            "prob_language": probabilities[1] * 100
        }
        return result

    except Exception as e:
        print(f"❌ 解析エラー: {e}")
        return None

    finally:
        # 一時ファイルのクリーンアップ
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

def main():
    # 引数から判定したいファイルパスを取得
    if len(sys.argv) < 2:
        print("💡 使用方法:")
        print("  python ./predict.py <mp4ファイルパス>")
        return

    target_file = sys.argv[1]
    print(f"🔍 判定中: {target_file}")
    
    result = predict_single_file(target_file)
    
    if result:
        print("\n==========================================")
        print(f"🎯 判定結果: 【 {result['prediction']} 】")
        print(f"==========================================")
        print(f" ── 🎵 音楽である確率 (Music)   : {result['prob_music']:.2f} %")
        print(f" ── 🗣️ 語学である確率 (Language): {result['prob_language']:.2f} %")
        print("==========================================\n")

if __name__ == "__main__":
    main()
