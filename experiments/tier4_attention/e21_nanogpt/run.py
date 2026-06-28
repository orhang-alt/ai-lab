"""e21 — nanoGPT-style char-level GPT (Tier 4: attention → a real language model).

The scaled-up, *actually trained* version of the GUI's "Tiny GPT" page. A real Transformer
language model in PyTorch: token + position embeddings, N blocks of **causal multi-head
self-attention** (softmax(QKᵀ/√d) with a triangular mask) + an MLP, each wrapped in
residual connections and LayerNorm; trained by **next-token cross-entropy** with AdamW;
runs on CPU / Apple-MPS / CUDA. Same ideas as the lab's GUI (Attention, Backprop,
Optimizers pages), now at scale.

Run:
    python experiments/tier4_attention/e21_nanogpt/run.py                 # offline fallback corpus
    python experiments/tier4_attention/e21_nanogpt/run.py --download      # tiny-shakespeare (~1 MB)
    python experiments/tier4_attention/e21_nanogpt/run.py --iters 5000 --n_layer 4 --n_embd 192

Needs PyTorch (`pip install torch`). On Python 3.14 it installs fine; if not, use a 3.12 venv.
"""

from __future__ import annotations

import argparse
import math
import os
import urllib.request

try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
except ModuleNotFoundError:
    raise SystemExit("PyTorch not installed. Run:  pip install torch   (3.14 works; else use a 3.12 venv)")

HERE = os.path.dirname(os.path.abspath(__file__))
TINY_SHAKESPEARE = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                    "master/data/tinyshakespeare/input.txt")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def fallback_corpus() -> str:
    """A self-contained, offline corpus of varied (original) English sentences."""
    import random
    rng = random.Random(0)
    subj = ["the cat", "a dog", "the child", "my friend", "the old man", "a small bird",
            "the river", "the teacher", "a sailor", "the queen", "the farmer", "a stranger"]
    verb = ["watched", "followed", "found", "carried", "painted", "remembered", "chased",
            "built", "opened", "lost", "wanted", "crossed", "loved", "feared", "carried"]
    obj = ["the small house", "a bright idea", "the long road", "an open book",
           "the quiet garden", "the dark forest", "a golden key", "the high mountain",
           "the deep sea", "a strange dream", "the empty street", "the morning light"]
    conj = ["and", "but", "so", "while", "because", "then", "yet"]
    lines = []
    for _ in range(1200):
        s = (f"{rng.choice(subj)} {rng.choice(verb)} {rng.choice(obj)} {rng.choice(conj)} "
             f"{rng.choice(subj)} {rng.choice(verb)} {rng.choice(obj)}.")
        lines.append(s[0].upper() + s[1:])
    return "\n".join(lines)


def load_text(args) -> str:
    path = os.path.join(HERE, "input.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    if args.download:
        print(f"downloading tiny-shakespeare (~1 MB) from\n  {TINY_SHAKESPEARE}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE, path)
        return open(path, encoding="utf-8").read()
    print("no input.txt and --download not set → using the built-in fallback corpus.\n"
          "(pass --download for tiny-shakespeare and far better samples.)")
    return fallback_corpus()


# --------------------------------------------------------------------------- #
# Model — a minimal GPT
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
        self.n_head = n_head
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)   # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)     # scaled dot product
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # causal
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual around attention
        x = x + self.mlp(self.ln2(x))    # residual around MLP
        return x


class GPT(nn.Module):
    def __init__(self, vocab, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, top_k=None):
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="nanoGPT-style char-level GPT")
    p.add_argument("--iters", type=int, default=200,
                   help="quick demo default; use 2000+ (with --download) for real samples")
    p.add_argument("--download", action="store_true", help="fetch tiny-shakespeare (~1MB)")
    p.add_argument("--device", default=None)
    p.add_argument("--n_layer", type=int, default=3)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gen", type=int, default=400, help="chars to generate at the end")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")

    text = load_text(args)
    chars = sorted(set(text))
    vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"device={device}  chars={len(text):,}  vocab={vocab}  "
          f"model: {args.n_layer}L/{args.n_head}H/{args.n_embd}d  block={args.block_size}")

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - args.block_size - 1, (args.batch,))
        x = torch.stack([d[i:i + args.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + args.block_size] for i in ix])
        return x.to(device), y.to(device)

    model = GPT(vocab, args.n_embd, args.n_head, args.n_layer, args.block_size, args.dropout).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"parameters: {nparams:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    @torch.no_grad()
    def estimate():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(20)
            for k in range(20):
                _, loss = model(*get_batch(split))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    eval_interval = max(1, args.iters // 4)
    for it in range(args.iters + 1):
        if it % eval_interval == 0 or it == args.iters:
            l = estimate()
            print(f"  step {it:5d}   train {l['train']:.3f}   val {l['val']:.3f}")
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    print("\nsample ------------------------------------------------------------")
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)   # start from char 0
    gen = model.generate(ctx, args.gen, temperature=0.8, top_k=min(20, vocab))[0].tolist()
    print("".join(itos[i] for i in gen))
    print("-------------------------------------------------------------------")
    print("\nThat's a real Transformer trained by next-token cross-entropy — the scaled-up "
          "version of the GUI's Tiny GPT. Use --download + more --iters/--n_layer for "
          "coherent English.")


if __name__ == "__main__":
    main()
