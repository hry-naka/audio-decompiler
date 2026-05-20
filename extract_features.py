import os
import shutil
import tempfile
import warnings
from pathlib import Path

# librosaのFutureWarningやUserWarningを非表示にする（インポート前に設定）
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import librosa
import numpy as np
import joblib
from tqdm import tqdm

# ==========================================
# 設定（学習・推論で共通化するパラメータ）
# ==========================================
DATASET_DIR = Path("./dataset")
SAVE_PATH = Path("./features_labels.joblib")

DURATION = 180       # 冒頭3分（180秒）を対象にする
SR = 16000           # サンプリングレートを16kHzに落としてCPU負荷軽減
N_MFCC = 20          # 特徴量の次元数

def get_mfcc_safely(file_path):
    """
    OneDriveの同期競合によるクラッシュを防ぐため、
    一度Macのローカル一時フォルダにファイルを安全にコピーしてから解析し、
    完了後に即座に削除する安全な関数
    """
    temp_dir = tempfile.gettempdir()
    temp_file_path = Path(temp_dir) / f"temp_analysis_{file_path.name}"
    
    try:
        # 1. 実体（OneDrive）をローカルの一時フォルダへ物理コピー
        real_src_path = file_path.resolve()
        shutil.copy2(real_src_path, temp_file_path)
        
        # 2. ローカルの一時ファイルから音声を読み込み
        y, sr = librosa.load(str(temp_file_path), duration=DURATION, sr=SR)
        
        if len(y) == 0:
            return None
            
        # 3. MFCCを計算
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        
        # 時間軸方向に平均値をとり、1次元（20個の数値）に圧縮
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
        
    except Exception as e:
        print(f"\n⚠️ スキップされました ({file_path.name}): {e}")
        return None
        
    finally:
        # 4. 用が済んだ一時ファイルは、成功・失敗に関わらず確実に消去
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

def main():
    X = []  # 特徴量（数字のリスト）を入れる箱
    y = []  # 正解ラベル（0: music, 1: language）を入れる箱

    # 分類カテゴリの定義
    categories = {
        "music": 0,
        "language": 1
    }

    print("🎙️ 音声特徴量（MFCC）の安全抽出を開始します...")

    for category_name, label in categories.items():
        folder_path = DATASET_DIR / category_name
        if not folder_path.exists():
            print(f"❌ フォルダが見つかりません: {folder_path}")
            continue

        # フォルダ内のmp4ファイル一覧を取得
        files = list(folder_path.glob("*.mp4")) + list(folder_path.glob("*.MP4"))
        print(f"\n📁 [{category_name}] から {len(files)} 個のファイルを処理中...")

        # tqdmで進捗バーを表示しながら処理
        for file_path in tqdm(files, desc=category_name):
            mfcc_features = get_mfcc_safely(file_path)
            
            if mfcc_features is not None:
                X.append(mfcc_features)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"\n✅ 特徴量抽出が完了しました！")
    print(f" ── 有効データ数: {len(X)} 個")
    print(f" ── 特徴量行列の形状 (データ数, 特徴量の次元): {X.shape}")

    # データを保存
    joblib.dump({'X': X, 'y': y}, SAVE_PATH)
    print(f"💾 安全にデータを {SAVE_PATH} に保存しました。")

if __name__ == "__main__":
    main()
