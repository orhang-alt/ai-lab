"""MLP — train a whole network (ANN module).

The payoff of the neuron + backprop pages: stack neurons into a hidden layer and a
multi-layer perceptron *learns* XOR — a problem one neuron can never solve. The Train tab
runs the lab's own Value MLP + optimizer live, showing the loss curve and the learned
decision boundary bending into shape.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # repo root (core)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap

import lessons
from core.nn import MLP
from core.optim import SGD, Adam, Momentum

# XOR with bipolar targets — tanh output in (-1, 1), predict by sign.
DATA = [([0, 0], -1.0), ([0, 1], 1.0), ([1, 0], 1.0), ([1, 1], -1.0)]


def _make_opt(name, params, lr):
    if name == "Momentum":
        return Momentum(params, lr=lr)
    if name == "Adam":
        return Adam(params, lr=lr)
    return SGD(params, lr=lr)


@st.cache_data(show_spinner=False)
def train(hidden: int, lr: float, opt_name: str, epochs: int, seed: int):
    model = MLP(2, [hidden, 1], nonlin="tanh", out_nonlin="tanh", seed=seed)
    opt = _make_opt(opt_name, model.parameters(), lr)
    losses = []
    for _ in range(epochs):
        loss = sum((model(x) - y) ** 2 for x, y in DATA) * (1.0 / len(DATA))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.data)
    preds = [model(x).data for x, _ in DATA]
    acc = sum(int((p >= 0) == (y >= 0)) for (_, y), p in zip(DATA, preds))
    g = np.linspace(-0.3, 1.3, 26)
    Z = [[model([float(gx), float(gy)]).data for gx in g] for gy in g]
    return losses, preds, acc, g.tolist(), Z


_ARCH_SVG = '''<div style="text-align:center;margin:0.5rem 0"><svg viewBox="0 0 460 250" style="width:100%;max-width:460px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A multilayer perceptron: two input nodes fully connected to a hidden layer of neurons, which connect to one output neuron."><rect x="1" y="1" width="458" height="248" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g stroke="#CBD7E6" stroke-width="1"><line x1="60" y1="95" x2="235" y2="55"/><line x1="60" y1="95" x2="235" y2="105"/><line x1="60" y1="95" x2="235" y2="155"/><line x1="60" y1="95" x2="235" y2="205"/><line x1="60" y1="160" x2="235" y2="55"/><line x1="60" y1="160" x2="235" y2="105"/><line x1="60" y1="160" x2="235" y2="155"/><line x1="60" y1="160" x2="235" y2="205"/></g><g stroke="#D9C7A8" stroke-width="1"><line x1="235" y1="55" x2="400" y2="130"/><line x1="235" y1="105" x2="400" y2="130"/><line x1="235" y1="155" x2="400" y2="130"/><line x1="235" y1="205" x2="400" y2="130"/></g><g fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.6"><circle cx="60" cy="95" r="18"/><circle cx="60" cy="160" r="18"/></g><g fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.6"><circle cx="235" cy="55" r="18"/><circle cx="235" cy="105" r="18"/><circle cx="235" cy="155" r="18"/><circle cx="235" cy="205" r="18"/></g><circle cx="400" cy="130" r="20" fill="#D7EFE5" stroke="#1D9E75" stroke-width="1.8"/><g font-family="sans-serif" font-size="11" fill="#33312E" text-anchor="middle"><text x="60" y="99">x₁</text><text x="60" y="164">x₂</text><text x="400" y="134">ŷ</text></g><g font-family="sans-serif" font-size="11" fill="#6B6A66" text-anchor="middle"><text x="60" y="210">inputs</text><text x="235" y="235">hidden layer (tanh)</text><text x="400" y="170">output</text></g></svg></div>'''

_LOOP_SVG = '''<div style="text-align:center;margin:0.5rem 0"><svg viewBox="0 0 560 170" style="width:100%;max-width:560px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The training loop: forward pass to predict, compute the loss, backward pass for gradients, optimizer step to update weights, then repeat."><defs><marker id="mlpl" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#5B8FC2"/></marker></defs><rect x="1" y="1" width="558" height="168" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g><rect x="20" y="48" width="115" height="46" rx="8" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.5"/><rect x="160" y="48" width="115" height="46" rx="8" fill="#FBEAF0" stroke="#C0507A" stroke-width="1.5"/><rect x="300" y="48" width="115" height="46" rx="8" fill="#FBEAD6" stroke="#9A6A2A" stroke-width="1.5"/><rect x="440" y="48" width="100" height="46" rx="8" fill="#D7EFE5" stroke="#1D9E75" stroke-width="1.5"/></g><g font-family="sans-serif" font-size="12" text-anchor="middle"><text x="77" y="68" fill="#0C447C">1. forward</text><text x="77" y="84" fill="#0C447C" font-size="10">predict ŷ</text><text x="217" y="68" fill="#8A2351">2. loss</text><text x="217" y="84" fill="#8A2351" font-size="10">how wrong?</text><text x="357" y="68" fill="#5A3E14">3. backward</text><text x="357" y="84" fill="#5A3E14" font-size="10">gradients</text><text x="490" y="68" fill="#0E5E45">4. step</text><text x="490" y="84" fill="#0E5E45" font-size="10">update w</text></g><g stroke="#5B8FC2" stroke-width="1.8" fill="none"><line x1="135" y1="71" x2="158" y2="71" marker-end="url(#mlpl)"/><line x1="275" y1="71" x2="298" y2="71" marker-end="url(#mlpl)"/><line x1="415" y1="71" x2="438" y2="71" marker-end="url(#mlpl)"/><path d="M490,94 C490,135 77,135 77,96" marker-end="url(#mlpl)"/></g><text x="283" y="130" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#6B6A66">repeat every step (backward = the Backprop page · step = Math X4)</text></svg></div>'''

_THEORY = r"""
## 1. From one neuron to a network

One neuron draws one straight line (ANN §3), so it can't solve **XOR**. Stack neurons into
a **hidden layer** and feed their outputs to an output neuron, and you get a **multilayer
perceptron (MLP)** — the hidden units invent new features that *bend* the input space until
the classes become separable.

<ARCH/>

## 2. Why depth + nonlinearity = a universal approximator

With a nonlinearity between layers (§8), a wide-enough MLP can approximate **any**
continuous function to arbitrary accuracy (the *universal approximation theorem*). Each
hidden unit contributes a "bump/fold"; combine enough of them and you can carve out any
region. Without the nonlinearity, stacked layers would collapse back to one line.

## 3. The training loop

Learning is the same four steps repeated until the loss is small:

<LOOP/>

1. **Forward** — run the inputs through to get predictions $\hat y$.
2. **Loss** — measure how wrong they are (here mean squared error).
3. **Backward** — `loss.backward()` fills every parameter's gradient (the **Backprop**
   page — autograd does this).
4. **Step** — the optimizer nudges each weight downhill, $w \leftarrow w - \eta\,\partial L/\partial w$ (Math **X4**).

The **Train** tab runs exactly this on the lab's `core.nn.MLP` + `core.optim` — watch the
loss curve fall and the decision boundary bend until XOR is solved 4/4.

## 4. The knobs

- **Hidden width** — more units = more capacity (can fit more complex boundaries, but can
  overfit, M0). Too few and XOR won't separate.
- **Learning rate $\eta$** — the X4 knob: too big diverges, too small crawls.
- **Optimizer** — plain **SGD**, **Momentum** (rolls through ravines), or **Adam**
  (adaptive per-parameter rates; often fastest to a good loss).
- **Seed** — the random initial weights; a bad init can get stuck (re-seed and retry).

## 5. What you're really watching

The hidden layer is **learning features**. Early on the boundary is a vague blob; as
gradients flow back and weights update, the hidden units rotate and shift their lines until
their combination cleanly fences off the two XOR corners. That — features learned from data
by gradient descent — is the whole idea behind deep learning. *(Lab: `core/nn.py`,
experiment e05.)*
"""

_QUIZ = [
    lessons.Question(
        "Why can an MLP solve XOR when a single neuron cannot?",
        ["it uses a bigger learning rate", "the hidden layer + nonlinearity bends input space so the classes become separable",
         "it has no bias", "it memorizes the four points"], 1,
        "Hidden units learn new features; with a nonlinearity their combination carves a non-linear boundary."),
    lessons.Question(
        "The four steps of the training loop, in order, are:",
        ["loss → forward → step → backward", "forward → loss → backward → step",
         "backward → step → forward → loss", "forward → backward → loss → step"], 1,
        "Predict (forward), measure error (loss), get gradients (backward), update weights (step) — then repeat."),
    lessons.Question(
        "What fills the parameters' gradients before the optimizer step?",
        ["the loss function alone", "loss.backward() — reverse-mode autograd",
         "the forward pass", "random initialization"], 1,
        "backward() back-propagates the loss to every parameter (the Backprop page); the optimizer then uses those grads."),
    lessons.Question(
        "Removing the nonlinearity between layers would…",
        ["make training faster with no downside", "collapse the whole stack to a single linear map (one line again)",
         "guarantee XOR is solved", "increase capacity"], 1,
        "Linear ∘ linear = linear, so depth adds no power without a nonlinearity (ANN §8)."),
]

_TASKS = r"""
### In the Train tab
1. Start at the defaults and confirm **4/4** on XOR; read the loss curve — where does it
   drop fastest?
2. Set **hidden = 2** — can it still solve XOR? Try a few **seeds**. (Two hidden units is
   the bare minimum; some inits get stuck.)
3. Crank the **learning rate** up until training **diverges**, then back off — find the
   edge.
4. Compare **SGD vs Adam** at the same lr/epochs: which reaches a low loss in fewer steps?

### Pencil & paper
5. How many parameters does an MLP `2 → h → 1` (tanh) have, as a function of $h$?
6. Sketch how two hidden lines can combine to fence off the two positive XOR corners.

### Code / bridge
7. Reproduce training with `core.nn.MLP` + `core.optim.SGD` in the Sandbox (this is e05).
8. Connect the dots: **Backprop** page (gradients) + **Math X4** (the step) = this loop.
"""

_REFS = r"""
- Nielsen — *Neural Networks and Deep Learning*, ch. 1–2 (MLPs + how they learn).
- 3Blue1Brown — *Neural networks* series (visual MLP intuition).
- Cybenko (1989) / Hornik (1991) — the **universal approximation** theorems.
- Goodfellow, Bengio & Courville — *Deep Learning*, ch. 6 (feedforward networks).
- In this lab: **Backprop** page (gradients), Math **X4** (optimization), **Two neurons**
  (the smallest XOR network), experiments **e05** (this demo) and **e07/e08**.
"""


st.title("MLP — train a whole network")
st.caption("Stack neurons into a hidden layer and a multilayer perceptron learns XOR — "
           "trained live on the lab's Value MLP + optimizer (core/nn.py, core/optim.py).")

tab_train, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🏋 Train it", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_train:
    st.markdown("A 2-input MLP with one hidden layer learns **XOR** (targets ±1, tanh output, "
                "predict by sign). Tune the network and watch it train:")
    c = st.columns(4)
    hidden = c[0].select_slider("hidden units", [2, 3, 4, 6, 8, 12], value=8, key="mlp_h")
    lr = c[1].select_slider("learning rate", [0.01, 0.03, 0.1, 0.3, 1.0], value=0.1, key="mlp_lr")
    opt_name = c[2].selectbox("optimizer", ["SGD", "Momentum", "Adam"], key="mlp_opt")
    epochs = c[3].select_slider("epochs", [50, 100, 200, 300, 500], value=300, key="mlp_ep")
    seed = st.slider("random seed (initial weights)", 0, 50, 1, key="mlp_seed")

    losses, preds, acc, g, Z = train(hidden, lr, opt_name, epochs, int(seed))

    m = st.columns(3)
    m[0].metric("final loss", f"{losses[-1]:.4f}")
    m[1].metric("XOR solved", f"{acc}/4", help="predictions correct by sign")
    m[2].metric("parameters", hidden * 4 + 1, help="2·h weights + h biases + h·1 weights + 1 bias")

    left, right = st.columns(2)
    with left:
        st.caption("Loss over training")
        st.line_chart(pd.DataFrame({"loss": losses}), height=260)
    with right:
        st.caption("Learned decision boundary")
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        G = np.array(g)
        xx, yy = np.meshgrid(G, G)
        ax.contourf(xx, yy, np.array(Z), levels=[-1.01, 0, 1.01],
                    colors=["#FAECE7", "#E6F1FB"], alpha=0.85)
        ax.contour(xx, yy, np.array(Z), [0], colors="k", linewidths=1.4)
        for (xv, yt), p in zip(DATA, preds):
            ax.scatter([xv[0]], [xv[1]], s=170, zorder=3,
                       color="#185FA5" if yt > 0 else "#A32D2D",
                       edgecolors="k", marker="o" if (p >= 0) == (yt > 0) else "X")
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.set_title("blue = +1, red = −1")
        st.pyplot(fig, width="stretch")

    df = pd.DataFrame({
        "x₁": [d[0][0] for d in DATA], "x₂": [d[0][1] for d in DATA],
        "target": [int(d[1]) for d in DATA],
        "output": [round(p, 3) for p in preds],
        "pred": [("+1" if p >= 0 else "−1") for p in preds],
    })
    st.dataframe(df, hide_index=True, width="content")
    if acc == 4:
        st.success("4/4 — the hidden layer bent input space until XOR became separable. "
                   "What one neuron (Two neurons page / e03) could never do.", icon=":material/check_circle:")
    else:
        st.warning(f"{acc}/4 — stuck in a poor minimum. Try a different **seed**, more "
                   "**hidden units**, or **Adam**.", icon=":material/refresh:")
    st.info("This is the real training loop on `core.nn.MLP` + `core.optim` — forward → MSE "
            "→ `loss.backward()` → `opt.step()`, repeated each epoch.", icon=":material/bolt:")

with tab_theory:
    body = _THEORY.replace("<ARCH/>", _ARCH_SVG).replace("<LOOP/>", _LOOP_SVG)
    st.markdown(body, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="mlp")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
