from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import PoetryDataset, collate_fn
from .model import PoetryLSTM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_sentences(path):
    text = path.read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

train_sentences = load_sentences(PROCESSED_DIR / "pretrain_train.txt")
val_sentences = load_sentences(PROCESSED_DIR / "pretrain_val.txt")
benchmark_sentences = load_sentences(PROCESSED_DIR / "pretrain_benchmark.txt")
with open(PROCESSED_DIR / "vocab_benchmark.json", "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)
vocab_size = len(char_to_idx)
print("train:", len(train_sentences))
print("val:", len(val_sentences))
print("vocab size:", vocab_size)
print()

train_dataset = PoetryDataset(train_sentences, char_to_idx)
val_dataset = PoetryDataset(val_sentences, char_to_idx)
benchmark_dataset = PoetryDataset(benchmark_sentences, char_to_idx)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn,
)
benchmark_loader = DataLoader(
    benchmark_dataset,
    batch_size=64,
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

criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx["<PAD>"])
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=4e-4,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1,
    min_lr=1e-5,
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_pretrained_model.pth"

pad_id = char_to_idx["<PAD>"]
bos_id = char_to_idx["<BOS>"]
eos_id = char_to_idx["<EOS>"]
unk_id = char_to_idx["<UNK>"]
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def generate_line(
    model,
    temperature=0.8,
    top_k=10,
    max_length=30,
):
    model.eval()
    token_id = bos_id
    hidden = None
    generated_chars = []
    with torch.no_grad():
        for _ in range(max_length):
            x = torch.tensor(
                [[token_id]],
                dtype=torch.long,
                device=device,
            )
            logits, hidden = model(x, hidden)
            next_logits = logits[0, -1]
            next_logits = next_logits / temperature
            next_logits[pad_id] = -float("inf")
            next_logits[bos_id] = -float("inf")
            next_logits[unk_id] = -float("inf")
            top_logits, top_indices = torch.topk(next_logits, k=top_k)
            top_probabilities = torch.softmax(top_logits, dim=-1)
            sampled_index = torch.multinomial(top_probabilities, num_samples=1).item()
            next_id = top_indices[sampled_index].item()
            if next_id == eos_id:
                break
            generated_chars.append(idx_to_char[next_id])
            token_id = next_id
    return "".join(generated_chars)

num_epochs = 50
best_val_loss = float("inf")

patience = 5
epochs_without_improvement = 0

for epoch in range(1, num_epochs + 1):
    model.train()
    total_train_loss = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
    train_loss = total_train_loss / len(train_loader)
    val_loss = evaluate(model, val_loader)
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch:02d} | "
        f"train loss: {train_loss:.4f} | "
        f"val loss: {val_loss:.4f} | "
        f"lr: {current_lr:.2e}"
    )
    sample = generate_line(
        model,
        temperature=0.7,
        top_k=15,
    )
    print(f"  sample: {sample if sample else '[空句]'}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("  -> saved best model")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping")
            break
print()

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)

print("===== Pretrain Samples =====")
for i in range(5):
    line_a = generate_line(
        model,
        temperature=0.7,
        top_k=15,
    )
    line_b = generate_line(
        model,
        temperature=0.7,
        top_k=15,
    )
    print()
    print(f"{i + 1}")
    print("A:", line_a)
    print("B:", line_b)

benchmark_loss = evaluate(model, benchmark_loader)
print(f"Benchmark loss: {benchmark_loss:.4f}")