import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors.torch import load_file
import json
import sys

# =========================
# CONFIG
# =========================

MODEL_PATH = "model.safetensors"
CONFIG_PATH = "config.json"
TOKENIZER_PATH = "tokenizer.model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD CONFIG
# =========================

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

block_size = config["block_size"]
n_embd = config["n_embd"]
n_head = config["n_head"]
n_layer = config["n_layer"]
vocab_size = config["vocab_size"]

# =========================
# LOAD TOKENIZER
# =========================

sys.path.append("minbpe")
from minbpe.basic import BasicTokenizer

tokenizer = BasicTokenizer()
tokenizer.load(TOKENIZER_PATH)

# =========================
# MODEL
# =========================

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2,-1) * (k.shape[-1] ** -0.5)

        weights = weights.masked_fill(
            self.tril[:T,:T] == 0,
            float("-inf")
        )

        weights = F.softmax(weights, dim=-1)

        v = self.value(x)

        return weights @ v


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        head_size = n_embd // n_head

        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(n_head)]
        )

        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)

        return self.proj(out)


class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
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

        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):

        B,T = idx.shape

        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T,device=device))

        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.head(x)

        return logits


model = GPTLanguageModel().to(device)

# =========================
# LOAD WEIGHTS
# =========================

state_dict = load_file(MODEL_PATH)
model.load_state_dict(state_dict)

model.eval()

# =========================
# GENERATION
# =========================

def generate(prompt, max_tokens=100, temperature=0.8, top_k=40):

    tokens = tokenizer.encode(prompt)

    tokens = torch.tensor(tokens, dtype=torch.long)[None,:].to(device)

    for _ in range(max_tokens):

        tokens_cond = tokens[:, -block_size:]

        logits = model(tokens_cond)

        logits = logits[:, -1, :] / temperature

        if top_k is not None:

            v, _ = torch.topk(logits, top_k)

            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat((tokens, next_token), dim=1)

    output = tokenizer.decode(tokens[0].tolist())

    return output

# =========================
# CHAT LOOP
# =========================

print("\nModel loaded. Type 'exit' to quit.\n")

while True:

    user = input("You: ")

    if user.lower() == "exit":
        break

    prompt = f"<speaker1> {user}\n<speaker2>"

    output = generate(prompt)

    reply = output.split("<speaker2>")[-1]

    print("AI:", reply.strip(), "\n")
