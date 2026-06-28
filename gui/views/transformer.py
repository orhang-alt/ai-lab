"""Tiny GPT — Transformer block & next-token prediction (ANN module).

The capstone: wrap attention in a Transformer block, stack blocks, and train on a single
objective — predict the next token (softmax over the vocabulary, cross-entropy). The Live
tab trains a tiny character-level language model and lets you prompt it and generate text
by sampling, with a temperature knob — exactly how an LLM produces words, in miniature.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neural_network import MLPClassifier

import lessons

CORPUS = (
    "the cat sat on the mat and the dog sat on the log. "
    "the cat ran in the sun and the dog ran in the fog. "
    "a cat and a dog sat in the sun on a mat by the log. "
    "the dog and the cat ran in the fog to the mat and the log. "
) * 3
K = 5   # context window (characters)


@st.cache_resource(show_spinner=False)
def train_lm():
    vocab = sorted(set(CORPUS))
    c2i = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)

    def enc(ctx):                       # k chars -> concatenated one-hots
        v = np.zeros(K * V)
        for j, ch in enumerate(ctx):
            v[j * V + c2i[ch]] = 1.0
        return v

    X, y = [], []
    for i in range(len(CORPUS) - K):
        X.append(enc(CORPUS[i:i + K])); y.append(c2i[CORPUS[i + K]])
    model = MLPClassifier(hidden_layer_sizes=(96,), max_iter=900, random_state=0)
    model.fit(np.array(X), np.array(y))
    return model, vocab, c2i, V, enc


def _context(prompt, vocab):
    ctx = [c for c in prompt.lower() if c in set(vocab)]
    ctx = ([" "] * K + ctx)[-K:]
    return "".join(ctx)


def _distribution(model, vocab, c2i, V, enc, ctx):
    p = model.predict_proba([enc(ctx)])[0]
    probs = np.zeros(len(vocab))
    for cls, pr in zip(model.classes_, p):
        probs[cls] = pr
    return probs


def _generate(model, vocab, c2i, V, enc, prompt, n, temp, rng):
    ctx = _context(prompt, vocab)
    out = ""
    for _ in range(n):
        probs = _distribution(model, vocab, c2i, V, enc, ctx)
        q = probs ** (1.0 / temp)
        q = q / q.sum()
        ch = vocab[rng.choice(len(vocab), p=q)]
        out += ch
        ctx = (ctx + ch)[-K:]
    return out


_BLOCK_SVG = '''<div style="text-align:center;margin:0.5rem 0"><svg viewBox="0 0 400 420" style="width:100%;max-width:380px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A Transformer block: input embeddings, then LayerNorm and causal self-attention with a residual add, then LayerNorm and a feed-forward MLP with a residual add; stacked N times, then a linear layer and softmax over the vocabulary giving next-token probabilities."><defs><marker id="tfa" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#9C9B95"/></marker></defs><rect x="1" y="1" width="398" height="418" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><rect x="58" y="66" width="284" height="244" rx="10" fill="none" stroke="#C9C8C1" stroke-dasharray="5 4"/><text x="330" y="60" text-anchor="end" font-family="sans-serif" font-size="11" fill="#6B6A66">Transformer block  × N</text><rect x="118" y="26" width="164" height="28" rx="6" fill="#E6F1FB" stroke="#5B8FC2"/><text x="200" y="45" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#0C447C">token + position embeddings</text><rect x="150" y="78" width="100" height="22" rx="5" fill="#FFFFFF" stroke="#C9C8C1"/><text x="200" y="93" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#6B6A66">LayerNorm</text><rect x="112" y="110" width="176" height="28" rx="6" fill="#FBEAF0" stroke="#C0507A"/><text x="200" y="129" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#8A2351">causal self-attention</text><circle cx="200" cy="158" r="11" fill="#FFFFFF" stroke="#9A6A2A"/><text x="200" y="162" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#9A6A2A">+</text><rect x="150" y="180" width="100" height="22" rx="5" fill="#FFFFFF" stroke="#C9C8C1"/><text x="200" y="195" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#6B6A66">LayerNorm</text><rect x="132" y="212" width="136" height="28" rx="6" fill="#FBEAD6" stroke="#9A6A2A"/><text x="200" y="231" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#5A3E14">feed-forward MLP</text><circle cx="200" cy="262" r="11" fill="#FFFFFF" stroke="#9A6A2A"/><text x="200" y="266" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#9A6A2A">+</text><g stroke="#9C9B95" stroke-width="1.6" fill="none"><line x1="200" y1="54" x2="200" y2="76" marker-end="url(#tfa)"/><line x1="200" y1="100" x2="200" y2="108" marker-end="url(#tfa)"/><line x1="200" y1="138" x2="200" y2="146" marker-end="url(#tfa)"/><line x1="200" y1="169" x2="200" y2="178" marker-end="url(#tfa)"/><line x1="200" y1="202" x2="200" y2="210" marker-end="url(#tfa)"/><line x1="200" y1="240" x2="200" y2="250" marker-end="url(#tfa)"/><path d="M74,70 V158 H187" marker-end="url(#tfa)"/><path d="M74,172 V262 H187" marker-end="url(#tfa)"/><line x1="200" y1="273" x2="200" y2="326" marker-end="url(#tfa)"/><line x1="200" y1="354" x2="200" y2="372" marker-end="url(#tfa)"/></g><text x="92" y="120" font-family="sans-serif" font-size="9" fill="#9C9B95" transform="rotate(-90 92 120)">residual</text><rect x="110" y="328" width="180" height="26" rx="6" fill="#FFFFFF" stroke="#5B8FC2"/><text x="200" y="345" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#0C447C">Linear → softmax (vocab)</text><rect x="120" y="374" width="160" height="28" rx="6" fill="#D7EFE5" stroke="#1D9E75"/><text x="200" y="392" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#0E5E45">P(next token)</text></svg></div>'''

_MASK_SVG = '''<div style="text-align:center;margin:0.5rem 0"><svg viewBox="0 0 300 240" style="width:100%;max-width:300px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A causal attention mask: a 5 by 5 grid where each row (query position) can attend only to columns at or before it — the lower-triangular cells are filled, the upper ones blocked."><rect x="1" y="1" width="298" height="238" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><text x="150" y="24" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#33312E">causal mask — attend to ≤ i only</text>''' + "".join(
    f'<rect x="{60+j*36}" y="{36+i*36}" width="34" height="34" '
    f'fill="{"#CFE3F5" if j<=i else "#EFEEE8"}" stroke="#FFFFFF"/>'
    + (f'<text x="{77+j*36}" y="{58+i*36}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#0C447C">✓</text>' if j <= i else "")
    for i in range(5) for j in range(5)
) + '''<text x="40" y="130" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#6B6A66" transform="rotate(-90 40 130)">query i →</text><text x="150" y="232" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#6B6A66">key j (position) →</text></svg></div>'''


_THEORY = r"""
## 1. From attention to a Transformer block

Attention (previous page) is the *mixer* that lets tokens share information. A **Transformer
block** wraps it into a reusable unit:

<BLOCK/>

Two ingredients beyond attention make deep stacks trainable:
- **Residual connections** (the `+`): each sub-layer adds to its input rather than
  replacing it, giving gradients a clean "highway" back through many layers (it's why the
  vanishing-gradient problem of ANN §6 doesn't kill deep Transformers).
- **LayerNorm**: re-centers/scales each token's vector so activations stay in a healthy
  range.

A block = **self-attention → add & norm → MLP → add & norm**. Stack $N$ of them and you
have the body of a GPT.

## 2. Causal masking — only look back

A language model predicts the **next** token, so when processing position $i$ it must not
peek at the future. **Causal (masked) self-attention** zeroes out attention to any position
$j > i$ — a lower-triangular mask:

<MASK/>

## 3. The output head — next-token softmax

The final token vectors go through a **linear layer → softmax over the vocabulary**, giving
$P(\text{next token}\mid\text{context})$. Training is a single objective: **maximize the
probability of the actual next token** = minimize **cross-entropy** (M2 / X5), over a huge
corpus. That's the *entire* learning signal — predict the next token, billions of times.

## 4. Generation = sampling, autoregressively

To produce text: feed the prompt, **sample** the next token from the softmax (a
**temperature** controls randomness — low = safe/repetitive, high = creative/chaotic; it's
the same softmax temperature from the Attention page, X5), append it, and repeat. The Live
tab does exactly this with a tiny **character-level** model.

## 5. That's a GPT

Stack masked-attention blocks, scale the data and parameters to billions, train on
next-token prediction — and you have a GPT. The whole lab built to here:
**neuron → MLP → backprop → optimizers → attention → Transformer → language model.**

> The Live demo is a fixed-window neural LM standing in for the stacked-attention version,
> but the **output head (next-token softmax)**, the **training objective
> (cross-entropy)**, and **generation (temperature sampling)** are exactly a GPT's.
"""

_QUIZ = [
    lessons.Question(
        "A Transformer block is, in essence:",
        ["just one big matrix multiply", "self-attention + a per-token MLP, each wrapped with a residual add and LayerNorm",
         "a convolution followed by pooling", "k-means over tokens"], 1,
        "Attention mixes tokens, the MLP processes each one; residuals + LayerNorm make deep stacks trainable."),
    lessons.Question(
        "What do residual connections buy a deep Transformer?",
        ["fewer parameters", "a gradient 'highway' that keeps deep stacks trainable (avoids vanishing)",
         "causal masking", "a larger vocabulary"], 1,
        "Adding to the input (instead of replacing it) lets gradients flow back through many layers — ANN §6."),
    lessons.Question(
        "Why the causal (triangular) mask?",
        ["to save memory", "so position i can't attend to future positions when predicting the next token",
         "to make attention symmetric", "to remove the softmax"], 1,
        "A next-token predictor must only use the past; the mask blocks attention to j > i."),
    lessons.Question(
        "How is a GPT trained, fundamentally?",
        ["clustering tokens", "predict the next token — minimize cross-entropy against the true next token, over a huge corpus",
         "matching images to labels", "k-fold cross-validation"], 1,
        "One self-supervised objective — next-token prediction (softmax + cross-entropy) — does it all."),
]

_TASKS = r"""
### In the Live tab
1. Type a prompt like `the ` and read the **next-character distribution** — which letters
   does the model expect, and do they match the corpus?
2. **Generate** at low **temperature** (≈0.4) vs high (≈1.3): low is repetitive/safe, high
   is wilder. Relate this to the softmax temperature from the Attention page (X5).
3. Find a temperature that produces the most "word-like" text. Why is there a sweet spot?

### Concept
4. Explain why a residual connection helps train a 50-layer Transformer (tie to ANN §6).
5. Sketch the causal mask for a length-4 sequence; which entries are zero and why?

### Bridge
6. Connect the dots end to end: **dot product (X1)** → **softmax (X5)** → **attention** →
   **block (residual + LayerNorm + MLP)** → **next-token cross-entropy (M2/X5)** = a GPT.
"""

_REFS = r"""
- Vaswani et al. (2017) — *Attention Is All You Need* (the Transformer).
- Radford et al. — *GPT / GPT-2* (decoder-only, next-token LM).
- Karpathy — *Let's build GPT* + **nanoGPT** (a real tiny GPT in code).
- Jay Alammar — *The Illustrated GPT-2*.
- In this lab: **Attention**, **MLP**, **Backprop**, Math **X5** (softmax/cross-entropy),
  roadmap **Tier 5**.
"""


st.title("Tiny GPT — Transformer & next-token prediction")
st.caption("Wrap attention in a block, stack it, and train on one objective: predict the "
           "next token. The Live tab trains a tiny char-level LM you can prompt and sample.")

tab_live, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["⌨ Generate", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_live:
    st.markdown("A tiny **character-level** language model trained on a few sentences — it "
                "predicts the **next character** from the last few. Real GPTs replace the "
                "fixed window with stacked attention, but the next-token softmax + sampling "
                "are the same.")
    with st.expander("training corpus (what it learned from)"):
        st.code(CORPUS[:220] + " …", language=None)

    model, vocab, c2i, V, enc = train_lm()

    prompt = st.text_input("prompt", value="the cat ", key="tf_prompt")
    cc = st.columns(2)
    temp = cc[0].slider("temperature", 0.3, 1.5, 0.7, 0.1, key="tf_temp",
                        help="low = safe/repetitive · high = random/creative")
    nchars = cc[1].slider("characters to generate", 20, 160, 80, key="tf_n")

    ctx = _context(prompt, vocab)
    probs = _distribution(model, vocab, c2i, V, enc, ctx)
    order = np.argsort(probs)[::-1][:12]
    labels = [("␣" if vocab[i] == " " else vocab[i]) for i in order]
    st.caption(f"Next-character probabilities after context “{ctx.replace(' ', '␣')}”")
    st.bar_chart(pd.DataFrame({"P(next)": probs[order]}, index=labels), height=220)

    if st.button("Generate ▶", key="tf_gen"):
        rng = np.random.default_rng()
        text = _generate(model, vocab, c2i, V, enc, prompt, nchars, temp, rng)
        st.markdown(f"**{prompt}**{text}")
    st.info("One objective — next-token prediction via softmax + cross-entropy — and "
            "generation is just sampling that softmax, token by token. Scale this up and "
            "it's a GPT.", icon=":material/smart_toy:")

with tab_theory:
    st.markdown(_THEORY.replace("<BLOCK/>", _BLOCK_SVG).replace("<MASK/>", _MASK_SVG),
                unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="transformer")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
