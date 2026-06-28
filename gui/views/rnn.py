"""RNN — recurrent networks for sequences (ANN module, roadmap e15 / Tier 3).

Sequences need memory. An RNN walks a sequence one step at a time, carrying a hidden
**state** that summarizes the past. The Live tab runs a one-unit RNN,
h_t = tanh(w_x·x_t + w_h·h_{t-1}), so you can watch the **recurrent weight w_h set how long
it remembers** — the seed of both long-range learning and the vanishing-gradient problem.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons


def _sequence(kind, n):
    x = np.zeros(n)
    if kind == "Impulse (one spike)":
        x[1] = 1.0
    elif kind == "Step (turns on)":
        x[n // 3:] = 1.0
    elif kind == "Alternating":
        x[::2] = 1.0
    else:  # Random
        x = (np.random.default_rng(0).random(n) < 0.3).astype(float)
    return x


def _run_rnn(x, wx, wh, b=0.0):
    h, hs = 0.0, []
    for xt in x:
        h = np.tanh(wx * xt + wh * h + b)
        hs.append(h)
    return np.array(hs)


_DIAG_SVG = '''<div style="text-align:center;margin:0.5rem 0"><svg viewBox="0 0 720 230" style="width:100%;max-width:720px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="An RNN drawn two ways: folded as one cell whose hidden state feeds back into itself (recurrent weight Wh), with input weight Wx and output weight Wy; and unrolled over time as a chain h0 to h3 where each step takes an input x_t, updates the state, and emits y_t."><defs><marker id="rnn" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#5B8FC2"/></marker></defs><rect x="1" y="1" width="718" height="228" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><text x="95" y="26" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#33312E">folded (one cell, reused)</text><circle cx="95" cy="120" r="26" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.8"/><text x="95" y="125" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#5A3E14">h</text><path d="M118,104 C160,80 160,160 118,136" fill="none" stroke="#9A6A2A" stroke-width="1.8" marker-end="url(#rnn)"/><text x="158" y="124" font-family="sans-serif" font-size="11" fill="#9A6A2A">Wₕ</text><line x1="95" y1="180" x2="95" y2="148" stroke="#5B8FC2" stroke-width="1.8" marker-end="url(#rnn)"/><text x="80" y="174" font-family="sans-serif" font-size="11" fill="#0C447C">Wₓ</text><text x="95" y="198" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#0C447C">x</text><line x1="95" y1="92" x2="95" y2="62" stroke="#1D9E75" stroke-width="1.8" marker-end="url(#rnn)"/><text x="80" y="84" font-family="sans-serif" font-size="11" fill="#0E5E45">Wy</text><text x="95" y="56" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#0E5E45">y</text><line x1="180" y1="120" x2="208" y2="120" stroke="#9C9B95" stroke-width="1.5" stroke-dasharray="4 3"/><text x="430" y="26" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#33312E">unrolled over time (same weights every step)</text><g font-family="sans-serif" font-size="13" text-anchor="middle"><circle cx="280" cy="120" r="22" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.6"/><text x="280" y="125" fill="#5A3E14">h₁</text><circle cx="380" cy="120" r="22" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.6"/><text x="380" y="125" fill="#5A3E14">h₂</text><circle cx="480" cy="120" r="22" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.6"/><text x="480" y="125" fill="#5A3E14">h₃</text><circle cx="580" cy="120" r="22" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.6"/><text x="580" y="125" fill="#5A3E14">h₄</text></g><g stroke="#9A6A2A" stroke-width="1.7" fill="none"><line x1="234" y1="120" x2="256" y2="120" marker-end="url(#rnn)"/><line x1="302" y1="120" x2="356" y2="120" marker-end="url(#rnn)"/><line x1="402" y1="120" x2="456" y2="120" marker-end="url(#rnn)"/><line x1="502" y1="120" x2="556" y2="120" marker-end="url(#rnn)"/></g><text x="234" y="112" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#9A6A2A">Wₕ</text><g stroke="#5B8FC2" stroke-width="1.6" fill="none"><line x1="280" y1="170" x2="280" y2="144" marker-end="url(#rnn)"/><line x1="380" y1="170" x2="380" y2="144" marker-end="url(#rnn)"/><line x1="480" y1="170" x2="480" y2="144" marker-end="url(#rnn)"/><line x1="580" y1="170" x2="580" y2="144" marker-end="url(#rnn)"/></g><g stroke="#1D9E75" stroke-width="1.6" fill="none"><line x1="280" y1="96" x2="280" y2="70" marker-end="url(#rnn)"/><line x1="380" y1="96" x2="380" y2="70" marker-end="url(#rnn)"/><line x1="480" y1="96" x2="480" y2="70" marker-end="url(#rnn)"/><line x1="580" y1="96" x2="580" y2="70" marker-end="url(#rnn)"/></g><g font-family="sans-serif" font-size="10" text-anchor="middle" fill="#0C447C"><text x="280" y="184">x₁</text><text x="380" y="184">x₂</text><text x="480" y="184">x₃</text><text x="580" y="184">x₄</text></g><g font-family="sans-serif" font-size="10" text-anchor="middle" fill="#0E5E45"><text x="280" y="64">y₁</text><text x="380" y="64">y₂</text><text x="480" y="64">y₃</text><text x="580" y="64">y₄</text></g><text x="430" y="212" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#9C9B95">the hidden state h carries information forward — the network's memory</text></svg></div>'''


_THEORY = r"""
## 1. Sequences need memory

MLPs and CNNs take a **fixed-size** input. But language, speech, music and time-series are
**variable-length** and **order matters** ("dog bites man" ≠ "man bites dog"). A **recurrent
neural network (RNN)** handles this by walking the sequence **one step at a time**, carrying
a hidden **state** vector that summarizes everything seen so far — its **memory**.

## 2. The recurrence

At each step the RNN folds the new input into the running state:
$$ h_t = \tanh\!\big(W_x\,x_t + W_h\,h_{t-1} + b\big),\qquad y_t = W_y\,h_t. $$
The **same** weights $W_x, W_h, W_y$ are used at *every* time step (**weight sharing across
time**, like a CNN shares a kernel across space). $h_{t-1}$ is what makes it recurrent: the
output depends on the whole history, not just the current input.

## 3. Unrolling & backprop-through-time

"Unroll" the loop across the sequence and an RNN becomes a **deep feed-forward network that
shares weights** between layers. You train it with ordinary backprop applied to that
unrolled graph — called **backprop-through-time (BPTT)**: gradients flow backward across the
time steps (Backprop page).

<DIAG/>

## 4. The recurrent weight is the memory knob

How long does information last? It's governed by $W_h$. Feed a single spike (the demo's
**impulse**) and watch the state:
- **small $W_h$** → the state forgets almost immediately (short memory);
- **$W_h \approx 1$** → it persists for many steps (long memory);
- **$W_h > 1$** → it can blow up (tanh caps it here, but gradients won't be capped).

Slide $w_h$ in the Live tab and see the impulse response stretch or vanish.

## 5. Vanishing & exploding through time

Backprop multiplies by $W_h$ at **every** step, so over a long sequence the gradient is
$W_h$ raised to a large power (Backprop §10). $|W_h|<1$ → it **vanishes** (the RNN can't
learn long-range dependencies — it forgets the start of the sentence); $|W_h|>1$ → it
**explodes**. This is the central weakness of plain RNNs.

## 6. LSTM & GRU — gated memory

The fix is **gates**. An **LSTM** adds a separate **cell state** (a protected "conveyor
belt") plus three learned gates — **forget** (what to drop), **input** (what to add), and
**output** (what to read). Because the cell state is carried with mostly *additive* updates,
gradients survive far longer, so LSTMs learn **long-range** structure. A **GRU** is a lighter
two-gate variant. These powered the first great results in translation and speech.

## 7. Why Transformers replaced them (mostly)

RNNs have two stubborn problems: they're **inherently sequential** (step $t$ needs step
$t{-}1$, so you can't parallelize over time → slow on GPUs), and even LSTMs strain at *very*
long range. **Attention / Transformers** (Attention page) fix both: they process **all
positions in parallel** and connect any two tokens **directly** in one step. That's why
modern NLP is Transformer-based. RNNs still shine where input arrives **streaming**, latency
is tight, or sequences are modest (some speech, control, and time-series tasks). *(Roadmap
e15; the real version is an LSTM char-model trained in PyTorch.)*
"""

_QUIZ = [
    lessons.Question(
        "What makes an RNN suited to sequences?",
        ["it has more layers", "it carries a hidden state that summarizes the past, updated step by step with shared weights",
         "it uses convolution", "it ignores order"], 1,
        "h_t = tanh(W_x x_t + W_h h_{t-1}) — the state is memory, and the same weights apply at every step."),
    lessons.Question(
        "In the demo, the recurrent weight w_h controls…",
        ["the input size", "how long information persists in the state (memory length)",
         "the number of classes", "the learning rate"], 1,
        "Small w_h forgets fast; w_h≈1 remembers long; w_h>1 blows up — it's the memory knob."),
    lessons.Question(
        "Why do plain RNNs struggle with long-range dependencies?",
        ["too few parameters", "gradients get multiplied by W_h each step, so they vanish or explode over long sequences",
         "they can't use tanh", "the inputs are too short"], 1,
        "Repeated multiplication across time (Backprop §10) shrinks or blows up gradients — the vanishing/exploding problem."),
    lessons.Question(
        "What do LSTM gates add over a plain RNN?",
        ["nothing", "a protected cell state + forget/input/output gates that preserve long-term memory",
         "convolution", "a softmax"], 1,
        "Gated, mostly-additive cell-state updates let gradients (and memory) survive far longer."),
    lessons.Question(
        "Transformers largely replaced RNNs because they…",
        ["are smaller", "process all positions in parallel and connect any two tokens directly (no sequential bottleneck)",
         "don't need data", "use no attention"], 1,
        "Parallelism + direct long-range connections beat the RNN's step-by-step state for most NLP."),
]

_TASKS = r"""
### In the Live tab
1. Use **Impulse** input. At **w_h = 0.2**, how many steps until the state ≈ 0? Now **w_h =
   0.9** — count again. That's *memory length* as a function of the recurrent weight.
2. Push **w_h > 1** with a **Step** input — watch the state pin to the tanh limits (and
   imagine the *gradient*, which tanh does **not** cap → exploding).
3. Set **w_h ≈ 0** — the state just copies the (scaled) input: no memory, an RNN reduced to
   a per-step neuron.

### Concept
4. Write the unrolled computation for a length-3 sequence and mark where the *same* W_h is
   reused — then explain why that causes vanishing/exploding gradients.
5. In one line each: what do the LSTM **forget**, **input**, and **output** gates do?
6. Give one task where you'd still pick an RNN/LSTM over a Transformer, and why.
"""

_REFS = r"""
- Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory (LSTM)*.
- Cho et al. (2014) — *GRU*. · Karpathy (2015) — *The Unreasonable Effectiveness of RNNs*.
- Olah — *Understanding LSTM Networks* (the classic gate walkthrough).
- Vaswani et al. (2017) — *Attention Is All You Need* (why RNNs were superseded).
- In this lab: **Backprop** (BPTT / vanishing gradients), **Attention** (the replacement),
  **Tiny GPT**.
"""


st.title("RNN — recurrent networks for sequences")
st.caption("Walk a sequence carrying a hidden state. The Live tab runs a 1-unit RNN so you "
           "can watch the recurrent weight set how long it remembers.")

tab_live, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🔁 Hidden state", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_live:
    st.markdown("A one-unit RNN: $h_t = \\tanh(w_x\\,x_t + w_h\\,h_{t-1})$. Pick an input "
                "sequence and tune the weights — watch the **hidden state** (memory) evolve:")
    c = st.columns(3)
    kind = c[0].selectbox("input sequence", ["Impulse (one spike)", "Step (turns on)",
                                             "Alternating", "Random"], key="rnn_in")
    wx = c[1].slider("input weight wₓ", 0.0, 3.0, 1.0, 0.1, key="rnn_wx")
    wh = c[2].slider("recurrent weight wₕ (memory)", 0.0, 1.3, 0.7, 0.05, key="rnn_wh")
    n = 18
    x = _sequence(kind, n)
    h = _run_rnn(x, wx, wh)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    t = np.arange(n)
    ax.bar(t, x, width=0.5, color="#CFE3F5", label="input $x_t$")
    ax.plot(t, h, "-o", color="#9A6A2A", lw=1.8, ms=4, label="hidden state $h_t$")
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xlabel("time step t"); ax.set_ylim(-1.1, 1.3); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("the state carries the past forward", fontsize=10)
    st.pyplot(fig, width="stretch")

    if kind.startswith("Impulse"):
        decayed = next((i - 1 for i in range(2, n) if abs(h[i]) < 0.1 * abs(h[1]) + 1e-9), n)
        st.metric("impulse remembered for ≈", f"{max(decayed-1,0)} steps",
                  help="how long after the spike the state stays above 10% — i.e. memory length")
    st.info("The recurrent weight wₕ is the **memory knob**: small forgets fast, ≈1 remembers "
            "long, >1 blows up. The same wₕ multiplies the gradient at every step too — which "
            "is exactly why long sequences vanish/explode, and why LSTMs add gates.",
            icon=":material/history:")

with tab_theory:
    st.markdown(_THEORY.replace("<DIAG/>", _DIAG_SVG), unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="rnn")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
