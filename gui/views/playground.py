import itertools
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import lessons
from core.neuron import Neuron  # the real building block — no duplication

LESSON = lessons.SINGLE_NEURON
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def sub(i):
    return str(i).translate(SUB)


_TRY_NEURON = '''import numpy as np
from core.neuron import Neuron

# Build a gate by hand, then check its truth table
gate = Neuron(2, activation="step", weights=[1, 1], bias=-1.5)  # AND
for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    print(x, "->", int(gate.forward(x)))
'''

_TRY_PERCEPTRON = '''import numpy as np
from core.neuron import Neuron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
y = np.array([0, 0, 0, 1])              # AND target (try OR: [0,1,1,1])
n = Neuron(2, activation="step", weights=[0.0, 0.0], bias=0.0)
lr = 0.1
for epoch in range(10):
    errors = 0
    for xi, yi in zip(X, y):
        pred = int(n.forward(xi))
        update = lr * (yi - pred)        # <-- the perceptron rule
        n.w += update * xi
        n.b += update
        errors += int(yi != pred)
    print(f"epoch {epoch}: errors={errors}")
    if errors == 0:
        break
print("learned:", n)
'''


st.title("Neuron — playground & lesson")
st.caption("e01 · a = φ(w·x + b).  One neuron, n inputs. Experiment, then learn the theory, test yourself, and practice.")

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

# --------------------------------------------------------------------------- #
# Playground — one interactive neuron with a selectable number of inputs
# --------------------------------------------------------------------------- #
with tab_play:
    n = st.segmented_control(
        "Inputs (n)", [1, 2, 3, 4], default=2,
        help="How many inputs this single neuron has — one weight per input.",
    ) or 2

    PRESETS = {  # only meaningful for the 2-input case
        "Custom": None,
        "AND (step)": dict(act="step", w=[1.0, 1.0], b=-1.5),
        "OR (step)": dict(act="step", w=[1.0, 1.0], b=-0.5),
        "NAND (step)": dict(act="step", w=[-1.0, -1.0], b=1.5),
        "AND (sigmoid)": dict(act="sigmoid", w=[4.0, 4.0], b=-6.0),
    }

    left, right = st.columns([0.42, 0.58])
    with left:
        p = {}
        if n == 2:
            p = PRESETS[st.selectbox("Preset", list(PRESETS))] or {}
        acts = ["sigmoid", "step", "tanh", "relu", "sign", "linear"]
        act = st.selectbox("activation φ", acts, index=acts.index(p.get("act", "sigmoid")))

        default_w = p.get("w", [4.0, 4.0] if n == 2 else [2.0] * n)
        weights = [
            st.slider(f"w{sub(i)}", -6.0, 6.0,
                      float(default_w[i] if i < len(default_w) else 1.0), 0.1)
            for i in range(n)
        ]
        b = st.slider("bias", -8.0, 8.0, float(p.get("b", -6.0 if n == 2 else -1.0)), 0.1)

        terms = " + ".join(rf"{weights[i]:.1f}\,x_{{{i}}}" for i in range(n))
        st.latex(rf"a = \varphi\big({terms} + ({b:.1f})\big)")

    neuron = Neuron(n, activation=act, weights=weights, bias=b)

    # All 2^n binary input combinations -> exact truth table for any n.
    corners = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
    corner_out = neuron.forward(corners)

    with right:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        if n == 1:
            xs = np.linspace(-0.5, 1.5, 300)
            ax.plot(xs, neuron.forward(xs.reshape(-1, 1)), color="#185FA5", lw=2)
            ax.axhline(0, color="0.85", lw=0.6)
            if abs(weights[0]) > 1e-9:
                xb = -b / weights[0]
                if -0.5 <= xb <= 1.5:
                    ax.axvline(xb, color="k", ls="--", lw=1.2, label="boundary z=0")
                    ax.legend(loc="best", fontsize=8)
            ax.scatter(corners[:, 0], corner_out, c=corner_out, cmap="RdBu",
                       edgecolors="k", s=160, zorder=3,
                       vmin=corner_out.min(), vmax=corner_out.max())
            ax.set_xlabel("x₀"); ax.set_ylabel("a"); ax.set_title("response  a(x₀)")
        else:
            gn = 200
            xs = np.linspace(-0.5, 1.5, gn)
            xx, yy = np.meshgrid(xs, xs)
            grid = np.c_[xx.ravel(), yy.ravel()]
            if n > 2:  # 2D slice through (x0, x1); the remaining inputs fixed at 0
                grid = np.c_[grid, np.zeros((grid.shape[0], n - 2))]
            surface = neuron.forward(grid).reshape(xx.shape)
            boundary = neuron.pre_activation(grid).reshape(xx.shape)

            cf = ax.contourf(xx, yy, surface, levels=20, cmap="RdBu", alpha=0.8)
            ax.contour(xx, yy, boundary, levels=[0], colors="k", linewidths=1.5)

            base = np.array(list(itertools.product([0, 1], repeat=2)), dtype=float)
            base_full = np.c_[base, np.zeros((4, n - 2))] if n > 2 else base
            base_out = neuron.forward(base_full)
            ax.scatter(base[:, 0], base[:, 1], c=base_out, cmap="RdBu",
                       edgecolors="k", s=170, zorder=3,
                       vmin=base_out.min(), vmax=base_out.max())
            for (x0v, x1v), o in zip(base, base_out):
                ax.annotate(f"{o:.2f}", (x0v, x1v), textcoords="offset points",
                            xytext=(10, 8), fontsize=9)

            ax.set_xlabel("x₀"); ax.set_ylabel("x₁")
            title = "response surface + boundary (w·x+b=0)"
            if n > 2:
                fixed = ", ".join(f"x{sub(i)}" for i in range(2, n))
                title += f"\n(2D slice: {fixed} = 0)"
            ax.set_title(title, fontsize=10)
            fig.colorbar(cf, ax=ax, shrink=0.8, label="a")
        st.pyplot(fig, width="stretch")

    st.subheader("Truth table")
    table = {f"x{sub(i)}": corners[:, i].astype(int) for i in range(n)}
    table["z = w·x + b"] = np.round(neuron.pre_activation(corners), 3)
    table["a = φ(z)"] = np.round(corner_out, 4)
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")

    if n == 1:
        st.caption("One input → the plot is the full response curve a(x₀); the dashed line is the threshold z=0.")
    elif n == 2:
        lessons.predict(
            "Load the **AND (step)** preset. Which **single** slider — moved which way — turns it "
            "into **OR**? And why does only that one change the gate?",
            "The **bias** `b`: drag it from −1.5 up toward −0.5. AND needs *both* inputs on to "
            "cross the threshold; OR fires when *either* is on, so it needs a lower bar → raise `b`. "
            "The **weights** set the boundary's orientation; the **bias** sets its offset — so `b` "
            "alone slides the same line from the AND position to the OR position without rotating "
            "it. (See Theory §2.)",
        )
        with st.expander("Why these gates (AND, OR, XOR …)?"):
            st.markdown(
                "Logic gates are the **smallest, exact tests** of what one neuron can do — "
                "4 points and a visible decision line.\n\n"
                "- **AND / OR / NAND / NOR** are *linearly separable* → a single neuron **can** do them.\n"
                "- **XOR / XNOR** are *not* separable → **no single neuron can** (with any weights). "
                "That failure is exactly why we need **hidden layers** (Tier 1).\n\n"
                "Full story in the **Theory** tab, §9."
            )
    else:
        st.caption(f"One neuron with **{n} inputs** → {2 ** n} input combinations (truth table is exact). "
                   "The plot is a 2D slice through (x₀, x₁) with the other inputs set to 0.")

# --------------------------------------------------------------------------- #
# Theory / Self-check / Tasks / References — from the lesson module
# --------------------------------------------------------------------------- #
with tab_theory:
    intro = getattr(LESSON, "intro_blocks", None)  # tolerate stale hot-reloads
    if intro:
        lessons.render_intro(intro)
    st.markdown(LESSON.theory, unsafe_allow_html=True)  # §3 embeds inline SVG diagrams

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading. Use it to find the gaps.")
    lessons.render_quiz(LESSON.quiz, prefix="neuron")

with tab_tasks:
    st.subheader("Tasks")
    if os.environ.get("AILAB_ENABLE_SANDBOX") == "1":  # only when the Sandbox page exists
        st.caption("Run a starter in the 🐍 Sandbox (numpy + core preloaded):")
        cta = st.columns(2)
        if cta[0].button("Try: single neuron →"):
            st.session_state["sandbox_code"] = _TRY_NEURON
            st.switch_page("views/sandbox.py")
        if cta[1].button("Try: perceptron rule →"):
            st.session_state["sandbox_code"] = _TRY_PERCEPTRON
            st.switch_page("views/sandbox.py")
    st.markdown(LESSON.tasks)

    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each task first, then check yourself — solo learning only works closed-loop.")
    lessons.solution(
        r"""**1.** Weights fix the boundary's *orientation*; the bias fixes its *offset*. AND and OR
share the same direction $\mathbf w=[1,1]$ and differ only in where the line sits, so moving $b$
alone slides AND → OR.

**2.** NAND $=\lnot$AND: flip the signs → $\mathbf w=[-1,-1],\,b=1.5$. NOR $=\lnot$OR:
$\mathbf w=[-1,-1],\,b=0.5$. Check: NAND$(1,1)=\text{step}(-2+1.5)=0$ ✓, NOR$(0,0)=\text{step}(0.5)=1$ ✓.

**3.** Scaling $\mathbf w,b$ by $k$ drives sigmoid → step. The tightest corners have $|z|=0.5k$;
for outputs within $0.01$ of $0/1$ you need $|z|\ge\ln 99\approx4.6$, so $k\gtrsim9.2$ — e.g.
$\mathbf w=[9,9],\,b=-13.5$ ($\lVert\mathbf w\rVert\approx13$).""",
        label="Warm-up 1–3",
    )
    lessons.solution(
        r"""**4.** Boundary $x_0+2x_1-3=0$: crosses the $x_0$-axis at $(3,0)$ and the $x_1$-axis at $(0,1.5)$.

**5.** Assume a separator exists. $(0,0){=}0\Rightarrow b<0$; the two 1-corners give $w_0+b>0$ and
$w_1+b>0$; $(1,1){=}0\Rightarrow w_0+w_1+b<0$. Add the two middle inequalities:
$w_0+w_1+2b>0\Rightarrow w_0+w_1+b>-b>0$ — contradicting $w_0+w_1+b<0$. So no line separates XOR.

**6.** **XOR** and **XNOR** — the only 2 of the 16 two-input boolean functions that are not linearly separable.

**7.** Augment $\tilde{\mathbf x}=(\mathbf x,1)$ and $\tilde{\mathbf w}=(\mathbf w,b)$. Then
$\tilde{\mathbf w}\cdot\tilde{\mathbf x}=\mathbf w\cdot\mathbf x+b$ — the bias is just a weight on a constant-1 input.

**8.** Signed distance $=(\mathbf w\cdot\mathbf x+b)/\lVert\mathbf w\rVert=(2+0-1)/\sqrt2=1/\sqrt2\approx0.707$.""",
        label="Pencil & paper 4–8",
    )
    lessons.solution(
        r"""**9.** (a) 30-layer hidden → **ReLU** (no saturation, cheap, avoids vanishing gradients).
(b) binary output → **sigmoid** (gives $P(y{=}1)$). (c) 10-class output → **softmax** (a distribution).
(d) output in $[-1,1]$ → **tanh**.

**10.** $2\sigma(2z)-1=\dfrac{2}{1+e^{-2z}}-1=\dfrac{1-e^{-2z}}{1+e^{-2z}}=\dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}=\tanh z$.

**11.** $\sigma'(z)=\sigma(z)\,(1-\sigma(z))$, maximal at $z=0$ where $\sigma=0.5$, giving $0.25$.
Backprop multiplies one $\sigma'\le0.25$ per layer, so 20 layers scale the gradient by
$\le0.25^{20}\approx10^{-12}$ — vanishing gradients, and the early layers barely move.""",
        label="Activations 9–11",
    )
    lessons.solution(
        r"""**12.** Add `signed_distance(self, x)` returning `(self.w @ x + self.b) / np.linalg.norm(self.w)`;
its test asserts $\approx0.707$ for $\mathbf w=[1,1],\,b=-1,\,\mathbf x=[2,0]$ (Task 8).

**13.** Brute-force integer $\mathbf w\in[-2,2]^2,\,b\in[-2,2]$ over the 4 corners — you'll find a
realization for every gate **except XOR / XNOR**.

**14.** Perceptron rule: `w += lr*(y-pred)*x; b += lr*(y-pred)`. It converges on AND/OR because they
are linearly separable (see experiment **e02**).

**15.** A single neuron shatters at most $n+1=3$ points in 2D (VC dimension $=3$): the separable
fraction stays $1.0$ for $k\le3$ and falls toward chance as $k\to4,5,6$.""",
        label="Code & stretch 12–15",
    )

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
