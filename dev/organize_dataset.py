import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ==========================================
# 設定
# ==========================================
SOURCE_BASE = Path("~/OneDrive/Sound/Radio/").expanduser()  # Chinese, English, Music があるフォルダ
TARGET_BASE = Path("./dataset")
SAMPLE_COUNT = 150  # 各ジャンル（語学/音楽）で最終的に集める合計数

LANGUAGE_SOURCES = [SOURCE_BASE / "Chinese", SOURCE_BASE / "English"]
MUSIC_SOURCES = [SOURCE_BASE / "Music"]


def collect_files_by_program(root_folders):
    """
    番組フォルダ（第1階層のサブフォルダ）ごとにmp4ファイルを分類して収集する
    返り値の構造: { 'ECN': [path1, path2, ...], 'SCN': [...] }
    """
    program_dict = defaultdict(list)
    
    for root in root_folders:
        if not root.exists():
            continue
        
        # 第1階層のフォルダ（番組フォルダ）をループ
        for program_dir in root.iterdir():
            if program_dir.is_dir():
                # その番組配下の全mp4を再帰的に取得
                mp4_files = list(program_dir.glob("**/*.mp4")) + list(program_dir.glob("**/*.MP4"))
                if mp4_files:
                    program_dict[program_dir.name].extend(mp4_files)
                    
    return program_dict


def sample_evenly(program_dict, total_required):
    """
    各番組から均等にファイルをつまみ食いして、指定の合計数（total_required）に達するまで集める
    """
    selected_files = []
    programs = list(program_dict.keys())
    
    if not programs:
        return selected_files
        
    # 各番組のファイルリストをあらかじめランダムシャッフルしておく
    for pg in programs:
        random.shuffle(program_dict[pg])
        
    # 全番組フォルダから1個ずつ順番に抜き取っていく（ラウンドロビン）
    while len(selected_files) < total_required:
        added_in_this_loop = 0
        
        for pg in programs:
            if len(selected_files) >= total_required:
                break
                
            # まだその番組に未抽出のファイルがあれば1個 pop する
            if program_dict[pg]:
                selected_files.append(program_dict[pg].pop())
                added_in_this_loop += 1
                
        # すべての番組フォルダが空っぽになったら終了
        if added_in_this_loop == 0:
            print(f"⚠️ 全ファイルを集め切りましたが、目標の {total_required} 個に届きませんでした（計 {len(selected_files)} 個）。")
            break
            
    return selected_files


def copy_files(file_list, target_dir):
    """ファイルをコピーする代わりに、シンボリックリンクを一瞬で作る"""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"🔗 {target_dir.name} へ {len(file_list)} 個のリンクを高速作成中...")
    
    for i, src_file in enumerate(file_list, 1):
        ext = src_file.suffix
        new_name = f"{target_dir.name}_{i:03d}{ext}"
        dst_file = target_dir / new_name
        
        # すでに古いリンクやファイルがある場合は一度消す（エラー防止）
        if dst_file.exists() or dst_file.is_symlink():
            dst_file.unlink()
            
        # 【ここを修正】実体コピーではなくシンボリックリンクを作成
        os.symlink(src_file.resolve(), dst_file)

def main():
    print("🔍 過去の録音ファイルを番組ごとにスキャン中...")

    # 1. 番組ごとにファイルを分類して収集
    lang_programs = collect_files_by_program(LANGUAGE_SOURCES)
    music_programs = collect_files_by_program(MUSIC_SOURCES)

    print(f" ── 語学の番組（フォルダ）数: {len(lang_programs)} 個")
    print(f" ── 音楽の番組（フォルダ）数: {len(music_programs)} 個")

    # 2. 均等サンプリングを実行
    print("\n⚖️ 番組ごとの配分を揃えてサンプリング中...")
    final_lang_files = sample_evenly(lang_programs, SAMPLE_COUNT)
    final_music_files = sample_evenly(music_programs, SAMPLE_COUNT)

    # 3. コピー処理
    print("\n🚀 コピーを開始します...")
    copy_files(final_lang_files, TARGET_BASE / "language")
    copy_files(final_music_files, TARGET_BASE / "music")

    print("\n✨ 完了しました！番組の偏りがない綺麗なデータセットが完成しました。")


if __name__ == "__main__":
    main()