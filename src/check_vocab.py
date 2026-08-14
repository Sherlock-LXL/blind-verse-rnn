from pathlib import Path
from collections import Counter
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_sentences(path):
    text = path.read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

def analyze_vocab(
    name,
    sentences,
    char_to_idx,
):
    char_counts = Counter()
    for sentence in sentences:
        char_counts.update(sentence)
    total_chars = sum(char_counts.values())
    unknown_counts = {
        char: count
        for char, count in char_counts.items()
        if char not in char_to_idx
    }
    unknown_total = sum(unknown_counts.values())
    unknown_unique = len(unknown_counts)
    unknown_rate = (
        unknown_total / total_chars
        if total_chars > 0
        else 0
    )
    print(f"===== {name} =====")
    print(f"Sentences: {len(sentences):,}")
    print(f"Total chars: {total_chars:,}")
    print(f"Unique chars: {len(char_counts):,}")
    print(f"Unknown chars: {unknown_total:,}")
    print(f"Unknown unique: {unknown_unique:,}")
    print(f"UNK rate: {unknown_rate:.4%}")
    print("\nTop 50 unknown chars:")
    for char, count in sorted(
        unknown_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:50]:
        print(f"{repr(char):8s} {count:6d}")
    print()
    return char_counts, unknown_counts

with open(
    PROCESSED_DIR / "vocab_benchmark.json",
    "r",
    encoding="utf-8",
) as f:
    char_to_idx = json.load(f)
print(f"Current vocab size: {len(char_to_idx):,}")
print()

datasets = {
    "Pretrain Train":
        PROCESSED_DIR / "pretrain_train.txt",
    "Pretrain Val":
        PROCESSED_DIR / "pretrain_val.txt",
    "Benchmark":
        PROCESSED_DIR / "pretrain_benchmark.txt",
    "Blind Train":
        PROCESSED_DIR / "train.txt",
    "Blind Val":
        PROCESSED_DIR / "val.txt",
    "Blind Test":
        PROCESSED_DIR / "test.txt",
}

all_pretrain_counts = Counter()
for name, path in datasets.items():
    if not path.exists():
        print(f"Skip {name}: {path.name} not found\n")
        continue
    sentences = load_sentences(path)
    char_counts, _ = analyze_vocab(
        name,
        sentences,
        char_to_idx,
    )
    if name == "Pretrain Train":
        all_pretrain_counts.update(char_counts)

print("===== Possible New Vocab =====")
special_tokens = 4
for min_count in [1, 2, 5, 10, 20, 50, 100]:
    kept_chars = sum(count >= min_count for count in all_pretrain_counts.values())
    print(f"min_count >= {min_count:3d}: vocab size = {kept_chars + special_tokens:,}")