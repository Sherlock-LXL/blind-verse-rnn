from pathlib import Path
from statistics import mean, median
import random
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
txt_files = list(RAW_DIR.rglob("*.txt"))
print(f"找到 {len(txt_files)} 个 txt 文件")
print("前 5 个文件：")
for path in txt_files[:5]:
    print(path)
print()

sentences = []
for path in txt_files:
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line:
            sentences.append(line)
print(f"原始非空句子数: {len(sentences)}")
print("前 5 个句子：")
for sentence in sentences[:5]:
    print(sentence)
print()

unique_sentences = []
seen = set()
for sentence in sentences:
    if sentence not in seen:
        unique_sentences.append(sentence)
        seen.add(sentence)
print(f"去重前: {len(sentences)}")
print(f"去重后: {len(unique_sentences)}")
print(f"删除重复: {len(sentences) - len(unique_sentences)}")
print()

lengths = [len(sentence) for sentence in unique_sentences]
print(f"最短句长: {min(lengths)}")
print(f"最长句长: {max(lengths)}")
print(f"平均句长: {mean(lengths):.2f}")
print(f"句长中位数: {median(lengths)}")
print(f"超过 20 字: {sum(length > 20 for length in lengths)}")
print(f"超过 30 字: {sum(length > 30 for length in lengths)}")
print(f"超过 40 字: {sum(length > 40 for length in lengths)}")
print()

special_tokens = ["<PAD>", "<BOS>", "<EOS>"]
all_chars = set()
for sentence in unique_sentences:
    for char in sentence:
        all_chars.add(char)
chars = sorted(all_chars)
vocab = special_tokens + chars
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
print(f"字符种类数: {len(chars)}")
print(f"包含特殊 token 后 vocab size: {len(vocab)}")
print()
print("前 10 个 vocab：")
for char, idx in list(char_to_idx.items())[:10]:
    print(idx, repr(char))
print()

example = unique_sentences[0]
encoded = ([char_to_idx["<BOS>"]] + [char_to_idx[char] for char in example] + [char_to_idx["<EOS>"]])
print("原句:")
print(example)
print("编码:")
print(encoded)
decoded = "".join(
    idx_to_char[idx]
    for idx in encoded
    if idx >= len(special_tokens)
)
print("解码:")
print(decoded)
print()

random.seed(42)
files = txt_files.copy()
random.shuffle(files)
num_files = len(files)
num_train = int(num_files * 0.8)
num_val = int(num_files * 0.1)
train_files = files[:num_train]
val_files = files[num_train:num_train + num_val]
test_files = files[num_train + num_val:]
print("文件划分：")
print("train:", len(train_files))
print("val:", len(val_files))
print("test:", len(test_files))
print()

def load_sentences(file_paths):
    sentences = []
    for path in file_paths:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences

train_sentences = load_sentences(train_files)
val_sentences = load_sentences(val_files)
test_sentences = load_sentences(test_files)
print("句子划分：")
print("train:", len(train_sentences))
print("val:", len(val_sentences))
print("test:", len(test_sentences))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def save_sentences(sentences, path):
    path.write_text("\n".join(sentences), encoding="utf-8")

save_sentences(train_sentences, PROCESSED_DIR / "train.txt")
save_sentences(val_sentences, PROCESSED_DIR / "val.txt")
save_sentences(test_sentences, PROCESSED_DIR / "test.txt")

with open(PROCESSED_DIR / "vocab.json", "w", encoding="utf-8") as f:
    json.dump(
        char_to_idx,
        f,
        ensure_ascii=False,
        indent=2,
    )