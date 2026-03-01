# =========================================================
# Install + Import minbpe
# =========================================================

!git clone https://github.com/karpathy/minbpe.git

import sys
sys.path.append("minbpe")

from minbpe.basic import BasicTokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file
import os, gc, json, time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# Stage 1 Hyperparameters
# =========================================================

block_size = 256
batch_size = 32
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.2
learning_rate = 3e-4
epochs = 2  # Stage 1 sanity

# =========================================================
# Load Dataset (FINAL CORRECT VERSION)
# =========================================================

DATA_FILE = "/kaggle/input/datasets/fasiowaizahmed/llm-training-90mb/sample.txt"

print("Loading dataset file...")

import os
assert os.path.exists(DATA_FILE), f"File not found: {DATA_FILE}"

with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read()

print("Corpus length (characters):", len(text))

# =========================================================
# Tokenizer
# =========================================================

from minbpe import BasicTokenizer

print("Training tokenizer...")
t0 = time.time()

tokenizer = BasicTokenizer()
tokenizer.train(text, vocab_size=2048)

print("Tokenizer training time:", round(time.time() - t0, 2), "seconds")

token_ids = tokenizer.encode(text)
vocab_size = len(tokenizer.vocab)

print("Vocab size:", vocab_size)
print("Token count:", len(token_ids))

# =========================================================
# Dataset
# =========================================================

class TokenDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.data = token_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

dataset = TokenDataset(token_ids, block_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# =========================================================
# Model Definition
# =========================================================

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        return weights @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

model = GPTLanguageModel().to(device)

# =========================================================
# Training
# =========================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting Stage 1 Training...")
t0 = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        del x, y, logits, loss
        torch.cuda.empty_cache()
        gc.collect()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

print("Training time:", round(time.time() - t0, 2), "seconds")

# =========================================================
# Save Outputs
# =========================================================

save_file(model.state_dict(), "/kaggle/working/model_stage1.safetensors")
tokenizer.save("/kaggle/working/tokenizer")

config = {
    "block_size": block_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "vocab_size": vocab_size
}

with open("/kaggle/working/config.json", "w") as f:
    json.dump(config, f)

print("✅ Stage 1 Complete")
