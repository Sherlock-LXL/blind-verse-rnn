from pathlib import Path
import json
import re
import random
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_DIR = PROJECT_ROOT / "data" / "pretrain"

dataset = load_dataset("xmj2002/Chinese_modern_classical", split="train")
dataset = dataset.shuffle(seed=42)

translation_sentences = []
seen = set()

target_chars = 500000
total_chars = 0

for row in dataset:
    text = row["modern"]
    parts = re.split(
        r"[，。！？；、,.!?;]",
        text
    )
    for part in parts:
        part = part.strip()
        if not re.fullmatch(r"[\u4e00-\u9fff]+", part):
            continue
        if not 5 <= len(part) <= 25:
            continue
        if part in seen:
            continue
        seen.add(part)
        translation_sentences.append(part)
        total_chars += len(part)
        if total_chars >= target_chars:
            break
    if total_chars >= target_chars:
        break

print(
    f"译文句子数: {len(translation_sentences)}"
)
print(
    f"译文字符数: {total_chars}"
)
print()

print("随机样例:")
for sentence in translation_sentences[:20]:
    print(sentence)

OUTPUT_PATH = PRETRAIN_DIR / "translation.txt"
OUTPUT_PATH.write_text("\n".join(translation_sentences), encoding="utf-8")
