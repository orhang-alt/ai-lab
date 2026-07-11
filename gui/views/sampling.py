"""Decoding / sampling strategies (ANN module, roadmap e23).

A trained language model outputs a probability for *every* next token. **Decoding** is how
you turn that distribution into an actual choice — and the same model writes very different
text depending on the strategy. This page takes one realistic next-token distribution and
lets you watch **greedy / temperature / top-k / top-p** reshape it and sample from it live.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons

# A realistic "The cat sat on the ___" next-token distribution: one strong choice,
# several plausible, a long implausible tail.
WORDS = ["mat", "floor", "sofa", "bed", "roof", "chair", "grass", "table",
         "windowsill", "fence", "idea", "purple", "banana"]
BASE = np.array([0.34, 0.18, 0.12, 0.09, 0.07, 0.05, 0.04, 0.03,
                 0.025, 0.02, 0.012, 0.008, 0.005])
BASE = BASE / BASE.sum()


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def decode(strategy, temp, k, p):
    """Return (probs_after_temperature, final_sampling_probs, kept_mask)."""
    q = _softmax(np.log(BASE + 1e-12) / temp)          # temperature reshaping
    if strategy == "Greedy (argmax)":
        out = np.zeros_like(q); out[int(q.argmax())] = 1.0
        mask = out > 0
        return q, out, mask
    mask = np.ones_like(q, dtype=bool)
    if strategy in ("Top-k", "Top-k + Top-p"):
        keep = np.argsort(q)[::-1][:k]
        mk = np.zeros_like(q, dtype=bool); mk[keep] = True
        mask &= mk
    if strategy in ("Top-p (nucleus)", "Top-k + Top-p"):
        order = np.argsort(q)[::-1]
        cum = np.cumsum(q[order])
        n = int(np.searchsorted(cum, p) + 1)           # smallest prefix reaching p
        mp = np.zeros_like(q, dtype=bool); mp[order[:n]] = True
        mask &= mp
    out = q * mask
    out = out / out.sum()
    return q, out, mask


_THEORY = r"""
## 1. The decoding problem

A trained language model gives you $P(\text{next token}\mid\text{context})$ — a probability
for **every** token in the vocabulary. **Decoding** (a.k.a. sampling) is the separate step
that turns that distribution into an actual chosen token. The model is fixed; the *strategy*
is a knob you control at generation time, and it changes the output dramatically — from
robotic and repetitive to fluent to gibberish.

## 2. Greedy / argmax

Always take the single most likely token. **Deterministic** and locally safe, but it tends
to be **repetitive and bland**, and "most likely token at each step" is **not** the most
likely *sentence* (a greedy path can paint itself into a corner). **Beam search** keeps the
best few partial sequences instead of one — better for short, "one right answer" tasks
(translation), still dull for open-ended writing.

## 3. Temperature

Reshape the distribution before choosing: divide the logits by $T$ (equivalently raise each
probability to $1/T$ and renormalize), then softmax (Math X5):
$$ p_i \;\propto\; \exp\!\big(\log p_i \,/\, T\big). $$
- $T \to 0$: collapses to **greedy** (sharpest).
- $T = 1$: the model's own distribution.
- $T > 1$: **flatter** → more random / "creative" / eventually chaotic.

Temperature alone still leaves a long tail of bad tokens with small-but-nonzero probability —
which is what the cutoffs below remove.

## 4. Top-k

Keep only the **$k$ most likely** tokens, set the rest to zero, renormalize, then sample.
Chops off the nonsense tail. The catch: a **fixed** $k$ is awkward — too small when the model
is genuinely unsure (many good options), too big when it's confident (one obvious answer).

## 5. Top-p (nucleus sampling)

Keep the **smallest set of tokens whose probabilities sum to ≥ $p$** (e.g. $p=0.9$), zero the
rest, renormalize, sample. This is **adaptive**: only a couple of tokens when the model is
confident, many when it's unsure. It's the usual default for open-ended generation, often
combined with a temperature and a generous top-k as a safety net.

## 6. Repetition control

Real decoders add a **repetition / frequency / presence penalty** that down-weights tokens
already used, to break the "the the the" loops that pure sampling can fall into.

## 7. Which to use

- **Factual / code / "one right answer"** → low temperature or greedy/beam.
- **Creative / chat** → temperature ≈ 0.7–1.0 **+** top-p ≈ 0.9.

There's no single correct setting — decoding is the dial on the **reliability ↔ creativity**
trade-off. *(Lab: the Tiny GPT page generates with temperature + top-k; this is roadmap
e23.)*
"""

_QUIZ = [
    lessons.Question(
        "What does the decoding strategy decide?",
        ["how the model is trained", "how to turn the model's next-token distribution into an actual chosen token",
         "the model's architecture", "the learning rate"], 1,
        "The model gives P(next token); decoding (greedy/temperature/top-k/top-p) picks from it at generation time."),
    lessons.Question(
        "Raising the temperature T does what?",
        ["sharpens toward the top token", "flattens the distribution → more random/creative output",
         "removes the softmax", "always improves accuracy"], 1,
        "T>1 flattens (more diverse), T<1 sharpens toward greedy, T→0 IS greedy."),
    lessons.Question(
        "Top-p (nucleus) sampling keeps…",
        ["a fixed number k of tokens", "the smallest set of tokens whose probabilities sum to at least p",
         "only the single best token", "all tokens equally"], 1,
        "Top-p is adaptive: few tokens when confident, many when unsure — unlike fixed top-k."),
    lessons.Question(
        "Greedy decoding is…",
        ["the same as top-p", "deterministic and often repetitive — and not necessarily the most likely sentence",
         "the most creative option", "random"], 1,
        "Argmax each step is deterministic and bland, and locally-greedy ≠ globally-best sequence."),
]

_TASKS = r"""
### In the Compare tab
1. Set **Greedy** — note it always returns the same token (`mat`). Switch to **Temperature**
   at T=1.0 and sample a few times — now you get variety.
2. Crank **temperature** up to ~1.6 — watch the implausible tail (`idea`, `purple`,
   `banana`) gain probability. Then add **Top-p = 0.9** — they vanish again.
3. Compare **Top-k = 3** vs **Top-p = 0.9** on this peaked distribution: which keeps fewer
   candidates? Now imagine a *flat* distribution — which adapts better?

### Concept
4. Explain why greedy decoding can be repetitive yet still not produce the most probable
   whole sentence.
5. When would you pick low temperature, and when high temperature + top-p?
"""

_REFS = r"""
- Holtzman et al. (2020) — *The Curious Case of Neural Text Degeneration* (introduces **top-p / nucleus**).
- Fan et al. (2018) — *Hierarchical Neural Story Generation* (**top-k**).
- Hugging Face — *How to generate text* (greedy/beam/temperature/top-k/top-p, with code).
- In this lab: **Tiny GPT** (generation), Math **X5** (softmax & temperature), **Attention**.
"""


st.title("Decoding — how an LLM picks the next token")
st.caption("One fixed next-token distribution, four strategies. Watch greedy / temperature / "
           "top-k / top-p reshape it and sample — the dial between reliable and creative.")

lessons.predict(
    'Same fixed next-token distribution, four decoders. Which one gives **identical** text every run, and which two **cut the tail** before sampling?',
    '**Greedy** always takes the argmax → identical, zero diversity. **top-k** and **top-p** cut the low-probability tail before sampling; **temperature** reshapes the whole distribution (higher = flatter / more random). Same model, very different text.',
)

tab_cmp, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎲 Compare", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_cmp:
    st.markdown("The model's next-token distribution after **“The cat sat on the ___”**. Pick "
                "a decoding strategy and see which tokens survive and what gets sampled:")
    c = st.columns(4)
    strategy = c[0].selectbox("strategy",
                              ["Greedy (argmax)", "Temperature", "Top-k", "Top-p (nucleus)",
                               "Top-k + Top-p"], index=1, key="sm_strat")
    temp = c[1].slider("temperature", 0.2, 2.0, 1.0, 0.1, key="sm_T",
                       disabled=strategy == "Greedy (argmax)")
    k = c[2].slider("k (top-k)", 1, len(WORDS), 4, key="sm_k",
                    disabled=strategy in ("Greedy (argmax)", "Temperature", "Top-p (nucleus)"))
    p = c[3].slider("p (top-p)", 0.1, 1.0, 0.9, 0.05, key="sm_p",
                    disabled=strategy in ("Greedy (argmax)", "Temperature", "Top-k"))

    q, out, mask = decode(strategy, temp, k, p)

    m = st.columns(3)
    m[0].metric("candidates kept", int(mask.sum()), help="tokens you can actually sample")
    m[1].metric("top candidate", WORDS[int(out.argmax())])
    m[2].metric("most-likely P", f"{out.max():.0%}")

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(WORDS))
    ax.bar(x - 0.2, q, width=0.4, color="#C9C8C1", label="after temperature")
    colors = ["#185FA5" if msk else "#EFEEE8" for msk in mask]
    ax.bar(x + 0.2, out, width=0.4, color=colors, edgecolor="#9C9B95", linewidth=0.4,
           label="final (sampled from)")
    ax.set_xticks(x); ax.set_xticklabels(WORDS, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("probability"); ax.legend(fontsize=8)
    ax.set_title(f"{strategy} — blue = kept, grey = cut", fontsize=10)
    st.pyplot(fig, width="stretch")

    if st.button("Sample 200 tokens ▶", key="sm_draw"):
        rng = np.random.default_rng()
        draws = rng.choice(len(WORDS), size=200, p=out)
        counts = np.bincount(draws, minlength=len(WORDS))
        top = np.argsort(counts)[::-1][:6]
        chips = "   ".join(f"`{WORDS[i]}` ×{counts[i]}" for i in top if counts[i] > 0)
        st.markdown("Drew 200 samples → " + chips)
        if strategy == "Greedy (argmax)":
            st.caption("Greedy always returns the same token — zero diversity.")

    st.info("Decoding is a generation-time choice, not part of the model. Temperature reshapes "
            "the whole distribution; top-k / top-p **cut the tail** before sampling. Same "
            "model, very different text.", icon=":material/casino:")

with tab_theory:
    st.markdown(_THEORY, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="sampling")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
