from pathlib import Path
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import PoetryDataset, collate_fn
from .model import PoetryLSTM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def load_sentences(path):
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


test_sentences = load_sentences(PROCESSED_DIR / "test.txt")
val_sentences = load_sentences(PROCESSED_DIR / "val.txt")

with open(PROCESSED_DIR / "vocab_benchmark.json", "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)

vocab_size = len(char_to_idx)
pad_id = char_to_idx["<PAD>"]
print("test:", len(test_sentences))
print("vocab size:", vocab_size)
print()

test_dataset = PoetryDataset(test_sentences, char_to_idx)
val_dataset = PoetryDataset(val_sentences, char_to_idx)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoetryLSTM(
    vocab_size=vocab_size,
    embedding_dim=1536,
    hidden_size=1536,
    num_layers=1,
    dropout=0.1,
).to(device)

MODEL_PATH = CHECKPOINT_DIR / "best_finetuned_model.pth"

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)

model.eval()
print("Loaded fine-tuned model.")
print()


def evaluate_exact(model, dataloader):
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
        reduction="sum",
    )
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            total_loss += loss.item()
            total_tokens += (targets != pad_id).sum().item()
    return total_loss / total_tokens


val_loss = evaluate_exact(model, val_loader)
test_loss = evaluate_exact(model, test_loader)
test_perplexity = math.exp(test_loss)

print(f"Val loss: {val_loss:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test perplexity: {test_perplexity:.2f}")
