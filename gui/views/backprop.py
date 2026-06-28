"""Backprop & autograd (ANN module).

The conceptual capstone: how a network gets a gradient for *every* weight in one
backward pass. The Live tab runs the lab's own scalar autograd engine
(`core.engine.Value`) on a tiny neuron + loss, showing the forward values and the
gradients it computes — then you can take one gradient-descent step and watch the loss drop.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # repo root (core)

import streamlit as st

import lessons
from core.engine import Value


def _compute(x0, w0, b0, yt):
    """Forward + backward on  L = (sigmoid(w*x + b) - y)^2  using the real engine."""
    x, w, b = Value(x0), Value(w0), Value(b0)
    z = w * x + b
    a = z.sigmoid()
    L = (a - yt) ** 2
    L.backward()
    return {
        "x": x.data, "w": w.data, "b": b.data, "z": z.data, "a": a.data, "L": L.data,
        "dx": x.grad, "dw": w.grad, "db": b.grad, "dz": z.grad, "da": a.grad,
    }


def _graph_svg(v: dict) -> str:
    return f'''<div style="text-align:center;margin:0.4rem 0"><svg viewBox="0 0 620 215" style="width:100%;max-width:620px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Live computational graph: inputs w, x, b feed z = w x + b, then a = sigmoid(z), then the loss L. Forward values flow right; the gradient of L with respect to each quantity flows back to the left."><defs><marker id="bpf" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#5B8FC2"/></marker><marker id="bpb" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#9A6A2A"/></marker></defs><rect x="1" y="1" width="618" height="213" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g stroke="#5B8FC2" stroke-width="1.8" fill="none"><line x1="114" y1="55" x2="220" y2="102" marker-end="url(#bpf)"/><line x1="114" y1="110" x2="220" y2="110" marker-end="url(#bpf)"/><line x1="114" y1="165" x2="220" y2="120" marker-end="url(#bpf)"/><line x1="280" y1="110" x2="360" y2="110" marker-end="url(#bpf)"/><line x1="420" y1="110" x2="488" y2="110" marker-end="url(#bpf)"/></g><g fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.4"><rect x="22" y="40" width="92" height="30" rx="6"/><rect x="22" y="95" width="92" height="30" rx="6"/><rect x="22" y="150" width="92" height="30" rx="6"/></g><g font-family="sans-serif" font-size="11.5" fill="#0C447C" text-anchor="middle"><text x="68" y="59">w = {v['w']:.2f}</text><text x="68" y="114">x = {v['x']:.2f}</text><text x="68" y="169">b = {v['b']:.2f}</text></g><g font-family="sans-serif" font-size="9.5" fill="#9A6A2A" text-anchor="middle"><text x="68" y="84">∂L/∂w = {v['dw']:.3f}</text><text x="68" y="139">∂L/∂x = {v['dx']:.3f}</text><text x="68" y="194">∂L/∂b = {v['db']:.3f}</text></g><circle cx="250" cy="110" r="28" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.8"/><circle cx="390" cy="110" r="28" fill="#D7EFE5" stroke="#1D9E75" stroke-width="1.8"/><circle cx="520" cy="110" r="30" fill="#FBEAF0" stroke="#C0507A" stroke-width="1.8"/><g font-family="sans-serif" text-anchor="middle"><text x="250" y="115" font-size="14" fill="#5A3E14">z</text><text x="390" y="115" font-size="15" fill="#0E5E45">σ</text><text x="520" y="115" font-size="14" fill="#8A2351">L</text></g><g font-family="sans-serif" font-size="10.5" fill="#33312E" text-anchor="middle"><text x="250" y="73">z = {v['z']:.2f}</text><text x="390" y="73">a = {v['a']:.3f}</text><text x="520" y="69">L = {v['L']:.3f}</text></g><g font-family="sans-serif" font-size="9.5" fill="#9A6A2A" text-anchor="middle"><text x="250" y="152">∂L/∂z = {v['dz']:.3f}</text><text x="390" y="152">∂L/∂a = {v['da']:.3f}</text></g><g font-family="sans-serif" font-size="10"><line x1="300" y1="190" x2="320" y2="190" stroke="#5B8FC2" stroke-width="2" marker-end="url(#bpf)"/><text x="324" y="194" fill="#5B8FC2">forward value</text><line x1="436" y1="190" x2="416" y2="190" stroke="#9A6A2A" stroke-width="2" marker-end="url(#bpb)"/><text x="440" y="194" fill="#9A6A2A">gradient (backward)</text></g></svg></div>'''


_CHAIN_SVG = '''<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 640 150" style="width:100%;max-width:640px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The chain rule along the path w to z to a to L: the gradient dL/dw is the product of the local derivatives dz/dw = x, da/dz = a(1-a), and dL/da = 2(a-y)."><defs><marker id="bch" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#9C9B95"/></marker></defs><rect x="1" y="1" width="638" height="148" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g fill="#FFFFFF" stroke="#5B8FC2" stroke-width="1.6"><circle cx="60" cy="60" r="22"/><circle cx="240" cy="60" r="22"/><circle cx="420" cy="60" r="22"/><circle cx="590" cy="60" r="22"/></g><g font-family="sans-serif" font-size="14" text-anchor="middle" fill="#0C447C"><text x="60" y="65">w</text><text x="240" y="65">z</text><text x="420" y="65">a</text><text x="590" y="65">L</text></g><g stroke="#9C9B95" stroke-width="1.6" fill="none"><line x1="84" y1="60" x2="214" y2="60" marker-end="url(#bch)"/><line x1="264" y1="60" x2="394" y2="60" marker-end="url(#bch)"/><line x1="444" y1="60" x2="564" y2="60" marker-end="url(#bch)"/></g><g font-family="sans-serif" font-size="11.5" text-anchor="middle" fill="#6B6A66"><text x="149" y="48">∂z/∂w = x</text><text x="329" y="48">∂a/∂z = a(1−a)</text><text x="505" y="48">∂L/∂a = 2(a−y)</text></g><text x="320" y="118" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#33312E">∂L/∂w  =  x · a(1−a) · 2(a−y)   — just multiply the local derivatives along the path</text></svg></div>'''


_THEORY = r"""
## 1. The goal — a gradient for every weight

Training is gradient descent (Math X4): repeatedly nudge each weight **against** its
gradient $\partial L/\partial w$. A real network has *millions* of weights, so the only
question that matters is: **how do we get $\partial L/\partial w$ for all of them, cheaply?**
The answer is **backpropagation** — and it computes *every* gradient in a single backward
sweep.

## 2. The forward pass

First run the network normally, left to right, **caching each intermediate value**. For one
sigmoid neuron with a squared-error loss:
$$ z = w\,x + b,\qquad a = \sigma(z),\qquad L = (a - y)^2. $$
Those cached numbers ($z, a, L$) are exactly what the backward pass needs.

## 3. The chain rule, backward

To get $\partial L/\partial w$, walk the path from $L$ back to $w$ and **multiply the local
derivatives** of each step (the chain rule):
$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial z}\cdot\frac{\partial z}{\partial w} = 2(a-y)\cdot a(1-a)\cdot x. $$

<CHAIN/>

Each factor is something the op already knows how to compute. The same backward value
$\partial L/\partial a$ is *reused* for $\partial L/\partial b$ and $\partial L/\partial x$
— that reuse is what makes backprop efficient.

## 4. Why "reverse" mode

There are two ways to apply the chain rule. **Forward mode** propagates the derivative of
*one input* through everything; **reverse mode** (backprop) propagates the derivative of
*one output* back to *everything*. ML has **one scalar loss** and **millions of
parameters**, so reverse mode — one backward pass gives *all* the gradients — is the
obvious win.

## 5. Every operation is local

Backprop needs no global calculus. Each operation only knows its **own** local derivative
(a tiny "send my gradient to my inputs" rule). To get all gradients you just:
1. topologically order the computational graph, then
2. apply the local rules in **reverse** order, accumulating into each node's `grad`.

That's literally all `core/engine.py` does — `+`, `*`, `**`, `sigmoid`, … each carry a
`_backward` rule, and `Value.backward()` runs them in reverse. **The Live tab is running
exactly that engine.**

## 6. Vanishing & exploding gradients

Because a gradient is a **product** of local derivatives along the path, depth multiplies
many factors together. If they're mostly $<1$ (e.g. sigmoid's $\le 0.25$), the product
**vanishes**; if mostly $>1$, it **explodes** (ANN §6). This is *why* ReLU, normalization,
and residual connections exist — to keep that product near 1 through many layers.

## 7. This is all "autograd"

PyTorch and TensorFlow do precisely this, just over **tensors** instead of scalars, on a
GPU. Once you've seen `Value.backward()` work on a tiny graph, you've seen the engine
behind every modern deep-learning framework. *(Lab: `core/engine.py`, experiments e04/e06.)*
"""

_QUIZ = [
    lessons.Question(
        "What does one backward pass of backprop produce?",
        ["the network's output", "the gradient of the loss w.r.t. every parameter",
         "the optimal weights directly", "the learning rate"], 1,
        "Backprop computes ∂L/∂(every weight) in a single reverse sweep; an optimizer then uses them to step."),
    lessons.Question(
        "Backprop is the chain rule applied how?",
        ["forwards, one input at a time", "backwards from the loss, multiplying local derivatives along each path",
         "by finite differences", "only to the last layer"], 1,
        "Reverse mode: start at the scalar loss and multiply local derivatives back toward each parameter."),
    lessons.Question(
        "Why reverse mode (not forward mode) for ML?",
        ["it's more accurate", "one scalar loss + many parameters → one backward pass gives all gradients",
         "it needs no cached values", "forward mode can't do the chain rule"], 1,
        "With one output and millions of inputs, reverse mode gets every gradient in a single pass."),
    lessons.Question(
        "Vanishing gradients happen because a path's gradient is…",
        ["a sum of large terms", "a product of many local derivatives, which shrinks if they're < 1",
         "independent of depth", "always exactly zero"], 1,
        "Chained products of sub-1 factors (e.g. sigmoid ≤ 0.25) decay toward 0 with depth — hence ReLU/normalization/residuals."),
]

_TASKS = r"""
### In the Live tab
1. Set the target far from the output and watch **∂L/∂w** grow; press **one GD step** and
   confirm the **loss drops**. Repeat — you're hand-training one neuron.
2. Find a setting where the output **saturates** (a near 0 or 1): notice **∂L/∂z = a(1−a)·…**
   collapses toward 0 — that's the vanishing-gradient effect, live.
3. Verify the chain by hand: check that `∂L/∂w` equals `x · a(1−a) · 2(a−y)`.

### Pencil & paper
4. Derive $\partial L/\partial b$ and $\partial L/\partial x$ for the same neuron. Which
   backward value do they share with $\partial L/\partial w$?
5. For a 2-layer net, write $\partial L/\partial w^{(1)}$ as a product of local derivatives.

### Code
6. Reproduce the Live example with `core.engine.Value` in the Sandbox; print every `.grad`.
7. Check the engine against finite differences: $\partial L/\partial w \approx (L(w+\epsilon)-L(w-\epsilon))/2\epsilon$ (this is experiment e06).
"""

_REFS = r"""
- **Karpathy — "The spelled-out intro to backpropagation" / micrograd** (the inspiration for `core/engine.py`).
- 3Blue1Brown — *Backpropagation* (visual intuition).
- Nielsen — *Neural Networks and Deep Learning*, ch. 2 (the four backprop equations).
- Goodfellow, Bengio & Courville — *Deep Learning*, §6.5.
- In this lab: Math **X2** (the chain rule), ANN **§6** (vanishing gradients), experiments **e04** (engine) and **e06** (gradient check).
"""


st.title("Backprop & autograd")
st.caption("How a network gets a gradient for every weight in one backward pass — run live "
           "on the lab's own Value engine (core/engine.py).")

tab_live, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🔁 Live walkthrough", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_live:
    st.markdown("One sigmoid neuron and a squared-error loss: "
                "$z=w x+b$, $a=\\sigma(z)$, $L=(a-y)^2$. Move the sliders — the engine "
                "recomputes the **forward values** and **back-propagates** the gradients:")
    c = st.columns(4)
    x0 = c[0].slider("input x", -3.0, 3.0, 1.0, 0.1, key="bp_x")
    w0 = c[1].slider("weight w", -3.0, 3.0, 0.8, 0.1, key="bp_w")
    b0 = c[2].slider("bias b", -3.0, 3.0, -0.2, 0.1, key="bp_b")
    yt = c[3].slider("target y", 0.0, 1.0, 1.0, 0.05, key="bp_y")

    v = _compute(x0, w0, b0, yt)
    st.markdown(_graph_svg(v), unsafe_allow_html=True)

    m = st.columns(3)
    m[0].metric("output a = σ(z)", f"{v['a']:.3f}")
    m[1].metric("loss L", f"{v['L']:.3f}")
    m[2].metric("∂L/∂w", f"{v['dw']:.3f}", help="gradient of the loss w.r.t. w, from Value.backward()")

    st.latex(
        rf"\frac{{\partial L}}{{\partial w}} = \underbrace{{2(a-y)}}_{{{2*(v['a']-yt):.3f}}}"
        rf"\cdot \underbrace{{a(1-a)}}_{{{v['a']*(1-v['a']):.3f}}}"
        rf"\cdot \underbrace{{x}}_{{{v['x']:.2f}}} = {v['dw']:.3f}"
    )

    if st.button("Take one gradient-descent step (η = 0.5)", key="bp_step"):
        lr = 0.5
        v2 = _compute(x0, w0 - lr * v["dw"], b0 - lr * v["db"], yt)
        delta = v2["L"] - v["L"]
        st.success(f"After one step: w {w0:.2f} → {w0 - lr*v['dw']:.2f}, "
                   f"b {b0:.2f} → {b0 - lr*v['db']:.2f}.  "
                   f"Loss {v['L']:.3f} → {v2['L']:.3f}  ({'−' if delta<0 else '+'}{abs(delta):.3f}).",
                   icon=":material/trending_down:")
    st.info("Every gradient above is computed by **`core.engine.Value.backward()`** — the "
            "same reverse-mode autograd that powers PyTorch, here on scalars.",
            icon=":material/bolt:")

with tab_theory:
    body = _THEORY.replace("<CHAIN/>", _CHAIN_SVG)
    st.markdown(body, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="backprop")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
