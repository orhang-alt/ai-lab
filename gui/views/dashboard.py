import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

import lab

st.title("🧠 AI Lab")
st.caption("From a single neuron to a small GPT — one growing library, many experiments.")

# --------------------------------------------------------------------------- #
# Learning roadmap — five levels, basics → frontier, with links to each page
# --------------------------------------------------------------------------- #
st.subheader("📚 Learning roadmap — a single neuron → a GPT")
st.markdown(
    "Five levels, **basics to frontier** — work top to bottom; each builds on the last and "
    "**clicking any step opens it**.\n\n"
    "The lab has **three tracks** (switch with the sidebar **Module** selector): **ANN** — "
    "this path, a single neuron → a GPT · **ML** — classical machine learning (M0–M8) · "
    "**Math** — the foundations (X1–X6) you *dip into as needed*. Each level flags the math "
    "it leans on."
)
st.page_link("views/the_chain.py", label="Start with the big picture — how it all connects",
             icon=":material/route:")

ROADMAP = [
    ("Level 1 · The neuron — foundations",
     "A neuron is just a **weighted sum + a nonlinearity** — one decision line. Meet it, watch "
     "**two** neurons already beat XOR, and see how neurons become **logic gates** and even do "
     "arithmetic. This is the atom everything else is built from.",
     [("views/playground.py", "Neuron playground", ":material/tune:"),
      ("views/two_neurons.py", "Two neurons", ":material/hub:"),
      ("views/neurons_compute.py", "Neurons → computer", ":material/calculate:")],
     "**X1** vectors & the dot product — a neuron *is* a dot product."),

    ("Level 2 · How a network learns — training",
     "The engine of deep learning: **backprop** finds a gradient for every weight, an "
     "**optimizer** steps downhill, and **regularization** stops it memorizing noise. Train "
     "your first network — it learns XOR live — and watch overfitting appear and get tamed.",
     [("views/backprop.py", "Backprop", ":material/sync_alt:"),
      ("views/mlp.py", "MLP (train it)", ":material/network_node:"),
      ("views/optimizers.py", "Optimizers", ":material/trending_down:"),
      ("views/deep_playground.py", "Deep nets (2D)", ":material/blur_on:"),
      ("views/regularization.py", "Regularization", ":material/tune:")],
     "**X2** calculus & the chain rule (backprop) · **X4** optimization (the optimizers)."),

    ("Level 3 · Architectures — images & sequences",
     "Wiring matched to the data: **convolutions** share a small filter across an image "
     "(**CNN**); **recurrence** carries a hidden state across a sequence (**RNN**). The RNN's "
     "memory limits are exactly what motivates attention next.",
     [("views/cnn.py", "CNN (images)", ":material/image:"),
      ("views/rnn.py", "RNN (sequences)", ":material/repeat:")],
     "**X1** dot products (convolution) · **X2** gradients through time (BPTT)."),

    ("Level 4 · Attention & Transformers",
     "The leap to modern AI: text becomes **tokens**, **self-attention** lets every token "
     "gather context by dot-product + softmax, and a stack of attention blocks predicts the "
     "**next token** — a tiny **GPT** you can run right here.",
     [("views/tokenization.py", "Tokenization", ":material/content_cut:"),
      ("views/attention.py", "Attention (LLMs)", ":material/auto_awesome:"),
      ("views/transformer.py", "Tiny GPT", ":material/smart_toy:")],
     "**X1** dot product = similarity · **X5** softmax."),

    ("Level 5 · LLMs in practice",
     "From a base model to a useful assistant: **decoding** turns probabilities into text, "
     "**embeddings + RAG** ground answers in facts, and **fine-tuning + RLHF** align "
     "behaviour. Then train the *real* thing — the **e21 nanoGPT** — from the Experiments page.",
     [("views/sampling.py", "Decoding (sampling)", ":material/casino:"),
      ("views/embeddings.py", "Embeddings & RAG", ":material/database:"),
      ("views/posttraining.py", "Post-training (RLHF)", ":material/psychology:"),
      ("views/experiments.py", "Experiments (e21 nanoGPT)", ":material/science:")],
     "**X5** cross-entropy & KL (training, decoding, alignment) · **X1** cosine similarity (RAG)."),
]

for title, intro, items, math in ROADMAP:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.markdown(intro)
        # lay the page links out in rows of up to 3
        for i in range(0, len(items), 3):
            row = items[i:i + 3]
            cols = st.columns(3)
            for col, (path, label, icon) in zip(cols, row):
                col.page_link(path, label=label, icon=icon)
        st.caption("➕ **Math you'll lean on:** " + math + "  *(Math module in the sidebar)*")

st.divider()

# --------------------------------------------------------------------------- #
# Build status — runs each experiment's run.py (done / todo / error)
# --------------------------------------------------------------------------- #
st.subheader("🔬 Experiment build status")
st.caption("Live status of the code experiments under `experiments/`. Heavy training demos are skipped by default.")


@st.cache_data(show_spinner="Scanning experiments…")
def scan(include_heavy: bool):
    """Run each experiment once; status = done / todo / error / heavy."""
    out = {}
    for tier, exps in lab.list_tiers().items():
        out[tier] = [(e, lab.status_of(e, include_heavy=include_heavy)) for e in exps]
    return out


ICON = {"done": "🟢", "todo": "⚪", "error": "🔴", "heavy": "⏱️"}

top = st.container()
controls = st.columns([0.7, 0.3])
include_heavy = controls[0].checkbox(
    "Include heavy training demos",
    value=False,
    help="Runs slower experiments such as e21 nanoGPT. Leave off for a quick dashboard scan.",
)
if controls[1].button("Re-scan", icon=":material/refresh:"):
    scan.clear()

data = scan(include_heavy)
flat = [s for rows in data.values() for _, s in rows]
done = flat.count("done")
heavy = flat.count("heavy")

with top:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experiments", len(flat))
    c2.metric("Done", done)
    c3.metric("Deferred", heavy)
    c4.metric("Remaining", len(flat) - done - heavy)
    st.progress(done / len(flat) if flat else 0.0)

for tier, rows in data.items():
    t_done = sum(1 for _, s in rows if s == "done")
    st.subheader(f"{lab.tier_label(tier)}  ·  {t_done}/{len(rows)}")
    for exp, status in rows:
        cols = st.columns([0.08, 0.5, 0.42])
        cols[0].write(ICON[status])
        cols[1].write(f"**{exp.id}** {exp.name}")
        cols[2].caption(status)

with st.expander("View ROADMAP.md"):
    st.markdown(lab.read(lab.LAB_ROOT / "ROADMAP.md"))
