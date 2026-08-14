from pathlib import Path
import random
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_DIR = PROJECT_ROOT / "data" / "pretrain"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

pretrain_files = [
    PRETRAIN_DIR / "modern.txt",
    PRETRAIN_DIR / "translation.txt",
    PRETRAIN_DIR / "song_ci.txt",
    PRETRAIN_DIR / "tang_shi.txt",
]

pretrain_sentences = []
seen = set()
raw_count = 0
for path in pretrain_files:
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        raw_count += 1
        if line in seen:
            continue
        seen.add(line)
        pretrain_sentences.append(line)

print(f"去重前: {raw_count}")
print(f"去重后: {len(pretrain_sentences)}")
print(f"删除重复: {raw_count - len(pretrain_sentences)}")
print()

total_chars = sum(len(sentence) for sentence in pretrain_sentences)
print(f"全部句子数: {len(pretrain_sentences)}")
print(f"全部字符数: {total_chars}")
print()

random.seed(42)
random.shuffle(pretrain_sentences)
val_ratio = 0.05
num_val = int(len(pretrain_sentences) * val_ratio)
pretrain_val = pretrain_sentences[:num_val]
pretrain_train = pretrain_sentences[num_val:]
print("Pretrain split:")
print("train:", len(pretrain_train))
print("val:", len(pretrain_val))
print()

print("随机样例:")
for sentence in pretrain_sentences[:20]:
    print(sentence)
print()

OUTPUT_PATH = PRETRAIN_DIR / "pretrain.txt"
OUTPUT_PATH.write_text("\n".join(pretrain_sentences), encoding="utf-8")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "pretrain_train.txt").write_text("\n".join(pretrain_train), encoding="utf-8")
(PROCESSED_DIR / "pretrain_val.txt").write_text("\n".join(pretrain_val), encoding="utf-8")

def load_sentences(path):
    text = path.read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

blind_train = load_sentences(PROCESSED_DIR / "train.txt")
blind_val = load_sentences(PROCESSED_DIR / "val.txt")
blind_test = load_sentences(PROCESSED_DIR / "test.txt")
all_sentences = pretrain_train + blind_train

special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
all_chars = set()
for sentence in all_sentences:
    for char in sentence:
        all_chars.add(char)
chars = sorted(all_chars)
vocab = special_tokens + chars
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
print(f"字符种类数: {len(chars)}")
print(f"包含特殊 token 后 vocab size: {len(vocab)}")
print()
print("前 10 个 vocab：")
for char, idx in list(char_to_idx.items())[:10]:
    print(idx, repr(char))
print()

def count_unknown(sentences, char_to_idx):
    unknown_count = 0
    total_count = 0
    unknown_chars = set()
    for sentence in sentences:
        for char in sentence:
            total_count += 1
            if char not in char_to_idx:
                unknown_count += 1
                unknown_chars.add(char)
    return unknown_count, total_count, unknown_chars

for name, sentences in [
    ("pretrain_val", pretrain_val),
    ("blind_val", blind_val),
    ("blind_test", blind_test),
]:
    unknown_count, total_count, unknown_chars = count_unknown(sentences, char_to_idx)
    print(name)
    print("unknown:", unknown_count)
    print("total:", total_count)
    print("unknown chars:", unknown_chars)
    print()

VOCAB_PATH = (PROCESSED_DIR / "vocab_pretrain.json")
with open(
    VOCAB_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        char_to_idx,
        f,
        ensure_ascii=False,
        indent=2,
    )