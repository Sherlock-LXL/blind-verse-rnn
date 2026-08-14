from pathlib import Path
import json
import re
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_DIR = PROJECT_ROOT / "data" / "pretrain"
POETRY_ROOT = PRETRAIN_DIR / "chinese-poetry-master"
SONG_DIR = POETRY_ROOT / "宋词"
song_files = sorted(SONG_DIR.glob("ci.song.*.json"))
print("宋词 json 文件数:", len(song_files))
print()

song_sentences = []
seen = set()
raw_count = 0
for path in song_files:
    with open(path, "r", encoding="utf-8") as f:
        poems = json.load(f)
    for poem in poems:
        for paragraph in poem["paragraphs"]:
            parts = re.split(r"[，。！？；、]", paragraph)
            for part in parts:
                part = part.strip()
                if not re.fullmatch(r"[\u4e00-\u9fff]+", part):
                    continue
                if not 5 <= len(part) <= 15:
                    continue
                raw_count += 1
                if part in seen:
                    continue
                seen.add(part)
                song_sentences.append(part)

print(f"去重前: {raw_count}")
print(f"去重后: {len(song_sentences)}")
print(f"删除重复: {raw_count - len(song_sentences)}")
print()

total_chars = sum(len(sentence) for sentence in song_sentences)
print(f"全部宋词字符数: {total_chars}")

random.seed(42)
random.shuffle(song_sentences)
target_chars = 900000
sampled_song = []
sampled_chars = 0

for sentence in song_sentences:
    sampled_song.append(sentence)
    sampled_chars += len(sentence)
    if sampled_chars >= target_chars:
        break

print(f"抽样句子数: {len(sampled_song)}")
print(f"抽样字符数: {sampled_chars}")
print()

print("随机样例:")
for sentence in sampled_song[:20]:
    print(sentence)

OUTPUT_PATH = PRETRAIN_DIR / "song_ci.txt"
OUTPUT_PATH.write_text("\n".join(sampled_song), encoding="utf-8")
