from pathlib import Path
import json
import re
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_DIR = PROJECT_ROOT / "data" / "pretrain"
POETRY_ROOT = PRETRAIN_DIR / "chinese-poetry-master"
TANG_DIR = POETRY_ROOT / "水墨唐诗"
song_files = sorted(TANG_DIR.glob("shuimotangshi.json"))
print("唐诗 json 文件数:", len(song_files))
print()

tang_sentences = []
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
                tang_sentences.append(part)

print(f"去重前: {raw_count}")
print(f"去重后: {len(tang_sentences)}")
print(f"删除重复: {raw_count - len(tang_sentences)}")
print()

total_chars = sum(len(sentence) for sentence in tang_sentences)
print(f"全部唐诗字符数: {total_chars}")

random.seed(42)
random.shuffle(tang_sentences)
target_chars = 30000
sampled_tang = []
sampled_chars = 0

for sentence in tang_sentences:
    sampled_tang.append(sentence)
    sampled_chars += len(sentence)
    if sampled_chars >= target_chars:
        break

print(f"抽样句子数: {len(sampled_tang)}")
print(f"抽样字符数: {sampled_chars}")
print()

print("随机样例:")
for sentence in sampled_tang[:20]:
    print(sentence)

OUTPUT_PATH = PRETRAIN_DIR / "tang_shi.txt"
OUTPUT_PATH.write_text("\n".join(sampled_tang), encoding="utf-8")
