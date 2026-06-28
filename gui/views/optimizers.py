"""Optimizers compared (ANN module).

SGD vs Momentum vs Adam, racing down an ill-conditioned "ravine" loss surface — run on
the lab's own core.optim, on a 2-D quadratic whose gradients flow through core.engine.Value.
Watch how momentum and Adam handle the steep/flat mismatch that makes plain SGD zig-zag.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # repo root (core)

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
from core.engine import Value
from core.optim import SGD, Adam, Momentum

START = (-4.5, -3.5)
OPTS = {"SGD": "#C0507A", "Momentum": "#5B8FC2", "Adam": "#1D9E75"}


def _make_opt(name, params, lr):
    if name == "Momentum":
        return Momentum(params, lr=lr, mu=0.9)
    if name == "Adam":
        return Adam(params, lr=lr)
    return SGD(params, lr=lr)


@st.cache_data(show_spinner=False)
def run(name: str, lr: float, steps: int, A: float):
    """Minimize f(x,y) = A·x² + y² with the real engine + optimizer; record the path."""
    x, y = Value(START[0]), Value(START[1])
    opt = _make_opt(name, [x, y], lr)
    xs, ys, losses = [x.data], [y.data], []
    for _ in range(steps):
        loss = A * x ** 2 + y ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        xs.append(x.data); ys.append(y.data); losses.append(loss.data)
        if abs(x.data) > 50 or abs(y.data) > 50:    # diverged — stop recording
            break
    return xs, ys, losses


_THEORY = r"""
## 1. The problem — ravines

Real loss surfaces are **ill-conditioned**: much steeper in some directions than others
(Math **X4 §10**). On such a surface the gradient doesn't point at the minimum — it points
mostly *across* a long narrow valley. Plain gradient descent then **overshoots the steep
direction and crawls along the flat one**, zig-zagging slowly. The demo's surface
$f(x,y)=A\,x^2+y^2$ is exactly this: steep in $x$ (large $A$), flat in $y$.

## 2. SGD — the baseline

$$ w \leftarrow w - \eta\,\nabla L. $$
Simple and memory-light. It works, but in a ravine it bounces across the steep axis while
inching down the flat one, and it can stall on **plateaus and saddle points**.

## 3. Momentum — a heavy ball

Accumulate a **velocity** that averages recent gradients, and step with *that*:
$$ v \leftarrow \mu\,v + \nabla L,\qquad w \leftarrow w - \eta\,v. $$
Consistent directions (down the valley) **build up speed**, while oscillating directions
(across the valley) **cancel out** — so momentum rolls through ravines far faster and
smooths the zig-zag. Typical $\mu \approx 0.9$.

## 4. Adam — adaptive per-parameter rates

Adam keeps a running mean $m$ (like momentum) **and** a running mean of squared gradients
$v$ (the gradient's size per parameter), then divides them:
$$ m \leftarrow \beta_1 m + (1-\beta_1)\nabla L,\quad v \leftarrow \beta_2 v + (1-\beta_2)(\nabla L)^2,\quad w \leftarrow w - \eta\,\frac{\hat m}{\sqrt{\hat v}+\varepsilon}. $$
Dividing by $\sqrt{\hat v}$ gives **each parameter its own effective step size** — big steps
where gradients are small (the flat $y$), small steps where they're large (the steep $x$).
That's why Adam takes a near-direct route with little tuning. It's the **robust default**,
especially for transformers/LLMs.

## 5. Which to use

- **Adam** — the safe default; fast, forgiving about the learning rate.
- **SGD + momentum** — often **generalizes best** for large vision nets (with a schedule).
- Whatever you pick, the **learning rate is still the #1 knob** (X4 §3) — too big diverges,
  too small crawls. Watch both happen in the demo. *(Lab: `core/optim.py`.)*
"""

_QUIZ = [
    lessons.Question(
        "Why does plain SGD zig-zag on an ill-conditioned (ravine) surface?",
        ["the learning rate is always too small", "the gradient points across the valley, so it overshoots the steep direction and crawls the flat one",
         "it has no bias term", "momentum is too high"], 1,
        "When curvature differs a lot between directions, −∇L doesn't aim at the minimum, so GD bounces."),
    lessons.Question(
        "What does momentum add to SGD?",
        ["a per-parameter learning rate", "a velocity that averages recent gradients, building speed down consistent directions",
         "second derivatives", "weight decay"], 1,
        "v ← μv + ∇L; w ← w − ηv. Consistent gradients accumulate; oscillations cancel."),
    lessons.Question(
        "Adam's key idea beyond momentum is to…",
        ["remove the learning rate", "divide by a running estimate of each gradient's size, giving per-parameter step sizes",
         "use the exact Hessian", "train without gradients"], 1,
        "Dividing m̂ by √v̂ adapts the step per parameter — big where grads are small, small where they're large."),
    lessons.Question(
        "Across all three optimizers, the single most important knob is still…",
        ["the random seed", "the learning rate η", "the batch size", "the number of parameters"], 1,
        "Too large diverges, too small crawls — true for SGD, Momentum and Adam alike (X4 §3)."),
]

_TASKS = r"""
### In the Compare tab
1. At the default learning rate, compare the three **paths** — which zig-zags, which rolls
   straight down the valley? Which reaches the minimum in the fewest steps (loss curve)?
2. Increase **ravine steepness A** — SGD's zig-zag gets worse; do momentum/Adam still cope?
3. Crank the **learning rate** up until **SGD diverges** (flies off the plot), then note
   that Adam often still survives. Now make it tiny — everything crawls.

### Pencil & paper
4. Write the momentum update and explain why it cancels oscillations but accelerates a
   steady descent.
5. In Adam, what happens to the effective step in a direction whose gradients are
   consistently **tiny**? Why does that help in a ravine?

### Code / bridge
6. Swap the optimizer in the **MLP** page (SGD ↔ Adam) and compare how fast XOR is solved.
7. Connect to Math **X4** (gradient descent, conditioning) and `core/optim.py` (these exact
   updates).
"""

_REFS = r"""
- Sebastian Ruder — *An overview of gradient descent optimization algorithms* (the classic survey).
- Kingma & Ba (2015) — *Adam*. · Sutskever et al. (2013) — *On the importance of momentum*.
- Goodfellow, Bengio & Courville — *Deep Learning*, ch. 8 (optimization for training).
- distill.pub — *Why Momentum Really Works* (interactive intuition).
- In this lab: Math **X4** (optimization), the **MLP** page (optimizer dropdown), `core/optim.py`.
"""


st.title("Optimizers compared — SGD · Momentum · Adam")
st.caption("The same ravine, three optimizers, run on the lab's core.optim. Watch how each "
           "handles a surface that's steep one way and flat the other.")

tab_cmp, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🏁 Compare", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_cmp:
    st.markdown("Minimizing the **ravine** $f(x,y) = A\\,x^2 + y^2$ (steep in $x$, flat in "
                "$y$) from the same start. Tune the shared learning rate and the steepness:")
    c = st.columns(3)
    lr = c[0].select_slider("learning rate η", [0.02, 0.05, 0.1, 0.2, 0.3], value=0.1, key="opt_lr")
    A = c[1].select_slider("ravine steepness A", [2, 4, 6, 9, 12], value=6, key="opt_A")
    steps = c[2].select_slider("steps", [30, 60, 100, 150], value=100, key="opt_steps")

    runs = {name: run(name, lr, steps, float(A)) for name in OPTS}

    left, right = st.columns(2)
    with left:
        st.caption("Paths over the loss contours")
        fig, ax = plt.subplots(figsize=(4.3, 3.9))
        gx = np.linspace(-5, 5, 80); gy = np.linspace(-5, 5, 80)
        XX, YY = np.meshgrid(gx, gy)
        ax.contour(XX, YY, A * XX ** 2 + YY ** 2, levels=12, colors="#D9D8D1", linewidths=0.8)
        for name, col in OPTS.items():
            xs, ys, _ = runs[name]
            ax.plot(xs, ys, "-o", color=col, ms=2.6, lw=1.4, label=name)
        ax.scatter([0], [0], marker="*", s=180, color="#33312E", zorder=5, label="min")
        ax.scatter([START[0]], [START[1]], s=40, color="#33312E", zorder=5)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_xlabel("x (steep)"); ax.set_ylabel("y (flat)")
        ax.legend(loc="upper right", fontsize=8)
        st.pyplot(fig, width="stretch")
    with right:
        st.caption("Loss vs step (log scale)")
        fig2, ax2 = plt.subplots(figsize=(4.3, 3.9))
        for name, col in OPTS.items():
            _, _, losses = runs[name]
            ax2.semilogy(np.maximum(losses, 1e-9), color=col, lw=1.8, label=name)
        ax2.set_xlabel("step"); ax2.set_ylabel("loss (log)")
        ax2.legend(loc="upper right", fontsize=8)
        st.pyplot(fig2, width="stretch")

    cols = st.columns(3)
    for (name, col), cc in zip(OPTS.items(), cols):
        _, _, losses = runs[name]
        final = losses[-1] if losses else float("nan")
        diverged = abs(runs[name][0][-1]) > 50 or abs(runs[name][1][-1]) > 50
        cc.metric(name, "diverged" if diverged else f"{final:.2e}",
                  help="final loss after the chosen steps")
    st.info("All three are the lab's `core.optim` (SGD/Momentum/Adam) stepping on gradients "
            "from `core.engine.Value.backward()`. Same η: SGD zig-zags down the steep axis, "
            "Momentum rolls through, Adam adapts per-axis.", icon=":material/bolt:")

with tab_theory:
    st.markdown(_THEORY, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="optim")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
