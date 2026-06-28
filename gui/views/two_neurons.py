"""Two-neuron playground (ANN module).

Explore how TWO neurons connect — and why that beats one. Toggle the wiring:
  • Parallel (a layer of 2): two independent lines → up to 4 regions / a 2-bit code.
  • Hidden → output: a hidden neuron + an output neuron that also sees the inputs
    (a skip) → two neurons solve XOR, which one neuron never can (e03).

Bridges e01 (one neuron) and e05 (the trained MLP). Reuses core.neuron.Neuron.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap

from core.neuron import Neuron

CORNERS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
GRID = np.linspace(-0.5, 1.5, 200)


def _grid():
    xx, yy = np.meshgrid(GRID, GRID)
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def parallel():
    st.markdown("**Two neurons, same inputs, no link** — each draws its own line; together "
                "they cut the plane into up to **4 regions** and give every point a 2-bit "
                "code (A, B). This is exactly a hidden *layer* of width 2.")
    left, right = st.columns([0.42, 0.58])
    with left:
        act = st.selectbox("activation φ", ["step", "sigmoid"], key="tn_p_act")
        st.markdown("**Neuron A** (coral line)")
        a0 = st.slider("wₐ₀", -4.0, 4.0, 1.0, 0.1, key="tn_a0")
        a1 = st.slider("wₐ₁", -4.0, 4.0, 0.0, 0.1, key="tn_a1")
        ab = st.slider("bₐ", -4.0, 4.0, -0.5, 0.1, key="tn_ab")
        st.markdown("**Neuron B** (blue line)")
        b0 = st.slider("w_b₀", -4.0, 4.0, 0.0, 0.1, key="tn_b0")
        b1 = st.slider("w_b₁", -4.0, 4.0, 1.0, 0.1, key="tn_b1")
        bb = st.slider("b_b", -4.0, 4.0, -0.5, 0.1, key="tn_bb")

    A = Neuron(2, activation=act, weights=[a0, a1], bias=ab)
    B = Neuron(2, activation=act, weights=[b0, b1], bias=bb)
    xx, yy, grid = _grid()
    a = (A.forward(grid) >= 0.5).astype(int)
    b = (B.forward(grid) >= 0.5).astype(int)
    code = (a * 2 + b).reshape(xx.shape)

    with right:
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        cmap = ListedColormap(["#FBEAF0", "#E1F5EE", "#FAEEDA", "#E6F1FB"])
        ax.contourf(xx, yy, code, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap=cmap, alpha=0.8)
        ax.contour(xx, yy, A.pre_activation(grid).reshape(xx.shape), [0], colors="#C0507A", linewidths=2)
        ax.contour(xx, yy, B.pre_activation(grid).reshape(xx.shape), [0], colors="#185FA5", linewidths=2)
        for (x0, x1) in CORNERS:
            ca, cb = int(A.forward([x0, x1]) >= 0.5), int(B.forward([x0, x1]) >= 0.5)
            ax.scatter([x0], [x1], s=120, color="k", zorder=4)
            ax.annotate(f"({ca},{cb})", (x0, x1), textcoords="offset points", xytext=(8, 6), fontsize=9)
        ax.set_xlabel("x₀"); ax.set_ylabel("x₁")
        ax.set_title("two lines → up to 4 regions")
        st.pyplot(fig, width="stretch")

    df = pd.DataFrame({
        "x₀": CORNERS[:, 0].astype(int), "x₁": CORNERS[:, 1].astype(int),
        "A": [int(A.forward(x) >= 0.5) for x in CORNERS],
        "B": [int(B.forward(x) >= 0.5) for x in CORNERS],
    })
    st.dataframe(df, hide_index=True, width="content")
    st.info("Two parallel neurons **transform** the input into a richer 2-D code, but can't "
            "yet make a single decision — something must combine A and B. That's the next "
            "wiring →", icon=":material/lightbulb:")


def hidden_output():
    st.markdown("**Hidden → output (with a skip).** Neuron *h* builds a feature; the output "
                "neuron *y* sees **the inputs and h**. With the defaults below, two neurons "
                "compute **XOR** — what one neuron never can (e03).")
    left, right = st.columns([0.42, 0.58])
    with left:
        st.markdown("**Hidden neuron h** = step(w·x + b)  *(default = AND)*")
        h0 = st.slider("w_h₀", -4.0, 4.0, 1.0, 0.1, key="tn_h0")
        h1 = st.slider("w_h₁", -4.0, 4.0, 1.0, 0.1, key="tn_h1")
        hb = st.slider("b_h", -4.0, 4.0, -1.5, 0.1, key="tn_hb")
        st.markdown("**Output y** = step(v₀x₀ + v₁x₁ + u·h + c)")
        v0 = st.slider("v₀", -4.0, 4.0, 1.0, 0.1, key="tn_v0")
        v1 = st.slider("v₁", -4.0, 4.0, 1.0, 0.1, key="tn_v1")
        u = st.slider("u  (weight on h)", -4.0, 4.0, -2.0, 0.1, key="tn_u")
        c = st.slider("c  (bias)", -4.0, 4.0, -0.5, 0.1, key="tn_c")

    h = Neuron(2, activation="step", weights=[h0, h1], bias=hb)
    y = Neuron(3, activation="step", weights=[v0, v1, u], bias=c)
    xx, yy, grid = _grid()
    hg = h.forward(grid)
    yg = y.forward(np.c_[grid, hg]).reshape(xx.shape)

    with right:
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        ax.contourf(xx, yy, yg, levels=[-0.5, 0.5, 1.5],
                    cmap=ListedColormap(["#FAECE7", "#E6F1FB"]), alpha=0.85)
        ax.contour(xx, yy, h.pre_activation(grid).reshape(xx.shape), [0],
                   colors="k", linewidths=1.6, linestyles="--")
        hc = h.forward(CORNERS)
        yc = y.forward(np.c_[CORNERS, hc])
        ax.scatter(CORNERS[:, 0], CORNERS[:, 1], c=yc, cmap=ListedColormap(["#A32D2D", "#185FA5"]),
                   s=150, edgecolors="k", zorder=4, vmin=0, vmax=1)
        for (x0, x1), yy_ in zip(CORNERS, yc):
            ax.annotate(f"{int(yy_)}", (x0, x1), textcoords="offset points", xytext=(9, 6), fontsize=10)
        ax.set_xlabel("x₀"); ax.set_ylabel("x₁")
        ax.set_title("output region (dashed = hidden line)")
        st.pyplot(fig, width="stretch")

    hc = h.forward(CORNERS).astype(int)
    yc = y.forward(np.c_[CORNERS, hc]).astype(int)
    df = pd.DataFrame({
        "x₀": CORNERS[:, 0].astype(int), "x₁": CORNERS[:, 1].astype(int),
        "h": hc, "y": yc,
    })
    st.dataframe(df, hide_index=True, width="content")
    is_xor = list(yc) == [0, 1, 1, 0]
    st.success("y = XOR ✓  — two neurons did what one couldn't." if is_xor
               else "Not XOR right now — reset the sliders to the defaults to see it.",
               icon=":material/check_circle:")
    st.info("The hidden **AND**-neuron fires only at (1,1); the output is basically **OR**, "
            "but `u = −2` lets h **veto** that corner → XOR. The hidden unit invents the "
            "feature that makes the problem linearly separable for the next neuron — the "
            "essence of hidden layers.", icon=":material/lightbulb:")


st.title("Two neurons — playground")
st.caption("How two neurons connect, and why that beats one. Toggle the wiring below.")

mode = st.segmented_control(
    "Wiring", ["Parallel (layer of 2)", "Hidden → output (XOR)"],
    default="Parallel (layer of 2)", key="tn_mode",
) or "Parallel (layer of 2)"
st.divider()

if mode.startswith("Parallel"):
    parallel()
else:
    hidden_output()
