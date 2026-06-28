"""The big picture — from one neuron to computing (ANN module).

A single page that ties the whole lab into one chain, with the two views (logic and
function) that meet in a trained network.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

_CHAIN_SVG = """
<svg viewBox="0 0 760 250" style="width:100%;height:auto;max-width:780px" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The chain from one neuron to an LLM">
  <defs>
    <marker id="chah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#5B8FC2"/>
    </marker>
  </defs>
  <rect x="1" y="1" width="758" height="248" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/>
  <text x="380" y="36" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#33312E">From a single neuron to computing</text>

  <g font-family="sans-serif">
    <!-- boxes -->
    <g fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.6">
      <rect x="18"  y="92" width="104" height="56" rx="8"/>
      <rect x="142" y="92" width="104" height="56" rx="8"/>
      <rect x="266" y="92" width="104" height="56" rx="8"/>
      <rect x="390" y="92" width="104" height="56" rx="8"/>
      <rect x="514" y="92" width="104" height="56" rx="8"/>
      <rect x="638" y="92" width="104" height="56" rx="8"/>
    </g>
    <g font-size="14" fill="#0C447C" text-anchor="middle">
      <text x="70"  y="124">1 neuron</text>
      <text x="194" y="124">logic gate</text>
      <text x="318" y="124">a layer</text>
      <text x="442" y="124">MLP</text>
      <text x="576" y="124">Transformer</text>
      <text x="690" y="124">LLM</text>
    </g>
    <g font-size="10.5" fill="#6B6A66" text-anchor="middle">
      <text x="70"  y="168">e01</text>
      <text x="194" y="168">gates · e02</text>
      <text x="318" y="168">two neurons</text>
      <text x="442" y="168">e05</text>
      <text x="576" y="168">Tier 3–4</text>
      <text x="690" y="168">Tier 5</text>
    </g>
    <!-- arrows -->
    <g stroke="#5B8FC2" stroke-width="2">
      <line x1="122" y1="120" x2="138" y2="120" marker-end="url(#chah)"/>
      <line x1="246" y1="120" x2="262" y2="120" marker-end="url(#chah)"/>
      <line x1="370" y1="120" x2="386" y2="120" marker-end="url(#chah)"/>
      <line x1="494" y1="120" x2="510" y2="120" marker-end="url(#chah)"/>
      <line x1="618" y1="120" x2="634" y2="120" marker-end="url(#chah)"/>
    </g>
  </g>

  <text x="380" y="216" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#6B6A66">every step = weighted sums (matrix multiplies) + a nonlinearity — and it's robust</text>
</svg>
"""

st.title("The big picture — from one neuron to computing")
st.caption("How everything in this lab links into one chain. Two views — logic and function "
           "— that meet in a trained network.")

st.markdown(f"<div style='text-align:center;margin:0.4rem 0'>{_CHAIN_SVG}</div>",
            unsafe_allow_html=True)

st.subheader("The chain, stop by stop")
st.markdown(r"""
1. **One neuron** — a weighted sum + threshold → one decision line / logic gate. *(Neuron playground, e01)*
2. **Logic gate / linear model** — AND/OR/NAND, or logistic regression: one neuron's whole repertoire. *(e01, e02; Neurons → computer)*
3. **A layer (many neurons)** — neurons in parallel build features; **two** already break the XOR wall and can **add numbers**. *(Two neurons; Neurons → computer)*
4. **MLP** — stacked layers + a nonlinearity = a **universal function approximator**, trained by backprop. *(e04–e05; Math X2 chain rule, X4 gradient descent)*
5. **Architectures** — add structure for the data: **CNN** (images), **RNN** (sequences/memory), **Transformer** (attention). *(roadmap Tier 3–4)*
6. **LLM** — billions of these neurons in Transformer blocks, predicting the next token via **softmax**. *(roadmap Tier 5)*
""")

st.subheader("Two views that meet")
st.markdown(r"""
- **Logic view:** neuron → gate → **circuit** → (with memory) a **universal computer**.
  NAND-completeness makes this exact (see *Neurons → computer*).
- **Function view:** neuron → linear model → **MLP** → fits **any function** → specialized
  to images/text.

A **trained network is both at once** — soft, *learned* logic gates **and** a function
approximator. The "learning" is just gradient descent (Math X4) finding the weights, and
backprop (Math X2's chain rule) supplying the gradients.
""")

st.subheader("What it physically is")
st.markdown(r"""
Every box above is the **same operation repeated**: a **weighted sum (matrix multiply) +
a nonlinearity**. That's literally what runs on a GPU; "depth" just means doing it many
times, each layer composing richer features from the last. And like a brain, it's
**robust** — no single weight is critical, because each decision rests on a sum of many.
""")

st.info("Where you are: you've built stops **1–4** for real (single neuron → autograd → an "
        "MLP that learns XOR). Stops **5–6** are the roadmap's Tier 3–5 — the path from here "
        "to a small GPT.", icon=":material/flag:")
