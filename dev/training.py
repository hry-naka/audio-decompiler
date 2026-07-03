import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 設定
# ==========================================
FEATURES_PATH = Path("./features_labels.joblib")
MODEL_SAVE_PATH = Path("./classifier.joblib")
SCALER_SAVE_PATH = Path("./scaler.joblib")

def main():
    if not FEATURES_PATH.exists():
        print(f"❌ 特徴量ファイルが見つかりません: {FEATURES_PATH}")
        print("先に extract_features.py を実行してください。")
        return

    print("📦 データの読み込み中...")
    data = joblib.load(FEATURES_PATH)
    X = data['X']
    y = data['y']

    print(f" ── データの数: {len(X)}")
    print(f" ── 特徴量の次元数: {X.shape[1]}")

    # ==========================================
    # 1. データを「学習用」と「テスト用」に分割
    # ==========================================
    # 全データ（300個）のうち 20%（60個）をテスト用にキープし、
    # 80%（240個）を使って学習します。偏りを防ぐため stratify（均等分割）を有効化。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n分割結果:")
    print(f" ── 学習データ: {len(X_train)} 個")
    print(f" ── テストデータ: {len(X_test)} 個")

    # ==========================================
    # 2. データの標準化（スケール調整）
    # ==========================================
    # MFCCの数字の大きさをAIが学習しやすい範囲（平均0、分散1）に整えます。
    # この時使ったスケール（Scaler）は、推論（Ubuntu）時にも使うので保存しておきます。
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # 3. AIモデル（SVM）の訓練
    # ==========================================
    print("\n🧠 AIモデル（サポートベクターマシン）をトレーニング中...")
    # 確率値（推論した時に「音楽である確率：95%」のような確率）を出力できるように probability=True にします
    model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    # ==========================================
    # 4. モデルの精度評価（テスト）
    # ==========================================
    print("\n📊 テストデータを使って精度を評価します...")
    y_pred = model.predict(X_test_scaled)

    # 正解率の算出
    accuracy = np.mean(y_pred == y_test)
    print(f"\n==========================================")
    print(f"🎉 テストデータに対する正解率 (Accuracy): {accuracy * 100:.2f} %")
    print(f"==========================================")

    # 詳細なスコアレポート
    # Precision（適合率）：音楽（または語学）と予測したもののうち、本当にそうだった確率
    # Recall（再現率）：本物の音楽（または語学）のうち、正しく予測できた確率
    print("\n📝 詳細レポート (0: 音楽, 1: 語学):")
    print(classification_report(y_test, y_pred, target_names=["Music", "Language"]))

    # 混同行列（どのジャンルをどれくらい間違えたかのマトリクス）
    cm = confusion_matrix(y_test, y_pred)
    print("🎯 混同行列 (Confusion Matrix):")
    print("                  予測: 音楽  予測: 語学")
    print(f"実際が 音楽:       {cm[0][0]:<10}{cm[0][1]}")
    print(f"実際が 語学:       {cm[1][0]:<10}{cm[1][1]}")

    # ==========================================
    # 5. モデルと標準化スケールの保存
    # ==========================================
    print("\n💾 成果物を保存します...")
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f" ── 学習済みモデルを保存しました: {MODEL_SAVE_PATH}")
    print(f" ── 標準化スケールを保存しました: {SCALER_SAVE_PATH}")
    print("\n✨ 学習プロセスがすべて完了しました！")

if __name__ == "__main__":
    main()
