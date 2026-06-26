import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

st.title("📐 Mathematical Basics")
st.caption("The math you actually need to study ML & ANN — same approach as the other "
           "modules: theory + interactive playground + self-check + tasks + references.")

st.info("The full X1–X6 math basics are built — pick any in the sidebar. X1 (vectors) and "
        "X4 (gradient descent) include interactive playgrounds.", icon=":material/rocket_launch:")

st.subheader("What you need to know")
st.markdown(r"""
- **X1 — Linear algebra** ✅ — vectors, the **dot product** (interactive), matrices as
  transformations, norms (L1/L2), projections, a peek at eigenvectors / SVD.
- **X2 — Calculus** ✅ — derivatives, **partial derivatives**, the **chain rule**,
  gradients, Jacobians, curvature. *(This is exactly what backpropagation is.)*
- **X3 — Probability & statistics** ✅ — distributions (esp. the **Gaussian**),
  expectation & variance, **Bayes' rule**, **maximum likelihood** (= where losses come from).
- **X4 — Optimization** ✅ — **gradient descent** (interactive), convexity, SGD,
  learning rate, momentum / Adam.
- **X5 — Information theory** ✅ — entropy, **cross-entropy**, KL divergence. *(The
  classification losses of ML/ANN.)*
- **X6 — Numerical computing** ✅ — floating point, numerical **stability** (log-sum-exp),
  vectorization, gradient checking.

**Suggested order:** X1 → X2 → X3 → X4, then X5 / X6 as needed.
""")

st.subheader("How the math powers the other modules")
st.markdown(r"""
| math topic | shows up as |
|---|---|
| dot product (X1) | a neuron's pre-activation $z=\mathbf w\cdot\mathbf x+b$ |
| matrix multiply (X1) | a whole layer $W\mathbf x+\mathbf b$ |
| chain rule (X2) | backpropagation (ANN Tier 1) |
| gradient descent (X4) | how every model learns |
| cross-entropy (X5) | the classification loss (ML & ANN) |
| Gaussian / MLE (X3) | why squared error = linear regression |

Switch modules anytime with the selector at the top of the sidebar.
""")
