from pathlib import Path
import json
import re
import random
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_DIR = PROJECT_ROOT / "data" / "pretrain"

dataset = load_dataset("Iess/chinese_modern_poetry", split="train")
dataset = dataset.shuffle(seed=42)

modern_sentences = []
seen = set()

target_chars = 4000000
total_chars = 0

for row in dataset:
    text = row["response"]
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("标题:") or line.startswith("标题："):
            continue
        line = re.sub(
            r"[^\u4e00-\u9fff]",
            "",
            line,
        )
        if not 5 <= len(line) <= 25:
            continue
        if line in seen:
            continue
        seen.add(line)
        modern_sentences.append(line)
        total_chars += len(line)
        if total_chars >= target_chars:
            break
    if total_chars >= target_chars:
        break

print(f"现代诗歌句子数: {len(modern_sentences)}")
print(f"现代诗歌字符数: {total_chars}")
print()

print("随机样例:")
for sentence in modern_sentences[:20]:
    print(sentence)

OUTPUT_PATH = PRETRAIN_DIR / "modern.txt"
OUTPUT_PATH.write_text("\n".join(modern_sentences), encoding="utf-8")
