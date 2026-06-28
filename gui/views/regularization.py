"""Regularization (ANN module).

The cure for the overfitting you can trigger on the Deep-nets page. Train a high-capacity
MLP on noisy data and dial **L2 regularization (weight decay)** up and down: watch the
decision boundary go from wiggly (memorizing noise) to smooth, and watch the train/test
gap close at the sweet spot — the bias–variance tradeoff (M0/M6), live.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lessons

ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def _data(seed):
    X, y = make_moons(n_samples=240, noise=0.30, random_state=seed)
    return train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)


def _model(alpha, seed):
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(40, 40), activation="relu",
                                       alpha=alpha, max_iter=2500, random_state=seed))


@st.cache_data(show_spinner=False)
def fit_alpha(alpha, seed):
    Xtr, Xte, ytr, yte = _data(seed)
    clf = _model(alpha, seed).fit(Xtr, ytr)
    allX = np.vstack([Xtr, Xte]); pad = 0.6
    gx = np.linspace(allX[:, 0].min() - pad, allX[:, 0].max() + pad, 70)
    gy = np.linspace(allX[:, 1].min() - pad, allX[:, 1].max() + pad, 70)
    xx, yy = np.meshgrid(gx, gy)
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    return {"Xtr": Xtr.tolist(), "ytr": ytr.tolist(), "Xte": Xte.tolist(), "yte": yte.tolist(),
            "gx": gx.tolist(), "gy": gy.tolist(), "Z": Z.tolist(),
            "train": clf.score(Xtr, ytr), "test": clf.score(Xte, yte)}


@st.cache_data(show_spinner=False)
def sweep(seed):
    Xtr, Xte, ytr, yte = _data(seed)
    tr, te = [], []
    for a in ALPHAS:
        clf = _model(a, seed).fit(Xtr, ytr)
        tr.append(clf.score(Xtr, ytr)); te.append(clf.score(Xte, yte))
    return tr, te


_THEORY = r"""
## 1. The disease — overfitting

A high-capacity network can drive **training** error to zero by memorizing the data, noise
and all — but then it fails on new data (**test** error stays high). That gap *is*
overfitting (M0): **low bias, high variance.**

## 2. The cure — penalize complexity

**Regularization** adds a penalty for being complex, so the model only uses capacity that
actually pays off. The most common is **L2 / weight decay** — add $\lambda\lVert\mathbf
w\rVert^2$ to the loss:
$$ \text{minimize}\quad L(\mathbf w) + \lambda\lVert\mathbf w\rVert^2. $$
Big weights are now expensive, so the network prefers **smaller weights → a smoother
function** that can't wiggle around every noisy point. $\lambda$ (sklearn's `alpha`) is the
strength dial.

## 3. The sweet spot (bias–variance, again)

- **Too little** $\lambda$ → wiggly boundary, big train–test gap (**overfit**).
- **Too much** $\lambda$ → boundary too smooth to capture the real shape (**underfit**).
- The best $\lambda$ — chosen on a **validation set** (M6) — sits in between, where test
  accuracy peaks. The **Tune** tab sweeps $\lambda$ and marks it.

## 4. Other ways to regularize a net

- **L1** — drives weights to exactly 0 (sparsity / feature selection).
- **Early stopping** — halt training when validation error starts rising.
- **Dropout** — randomly zero units during training so the net can't rely on any one.
- **More data / augmentation** — the most reliable cure of all.

All trade a little training fit for a lot more **generalization** — the whole game (M0/M6,
and "weight decay" in the ANN module). *(Lab tie-ins: Deep-nets page, M0, M6.)*
"""

_QUIZ = [
    lessons.Question(
        "Overfitting shows up as…",
        ["high train error and high test error", "low train error but high test error (a big gap)",
         "low test error only", "identical train and test error"], 1,
        "The model memorizes training noise: train error ↓ while test error stays high — high variance."),
    lessons.Question(
        "L2 regularization (weight decay) works by…",
        ["adding more layers", "penalizing large weights, so the model prefers a smoother function",
         "increasing the learning rate", "removing the test set"], 1,
        "Adding λ‖w‖² makes big weights costly → smaller weights → smoother fit that ignores noise."),
    lessons.Question(
        "As the regularization strength λ increases from tiny to huge, test accuracy…",
        ["always increases", "rises to a peak (the sweet spot) then falls as the model underfits",
         "always decreases", "stays flat"], 1,
        "Too little → overfit, too much → underfit; the best λ is in between (validation picks it)."),
]

_TASKS = r"""
### In the Tune tab
1. Slide $\lambda$ to its **smallest** value — describe the boundary and the train–test
   gap. Now slide to the **largest** — what happened to the boundary, and to *both*
   accuracies?
2. Find the $\lambda$ that maximizes **test** accuracy. Does it match the peak in the
   sweep curve?
3. Re-roll the **seed** — does the sweet-spot $\lambda$ move a lot? (That's why you tune it
   on validation, not guess.)

### Bridge
4. Connect L2 here to **ridge** (ML M1/M6), to **weight decay** in the ANN module, and to
   the bias–variance picture in **M0**.
5. Name two other regularizers you'd reach for in a deep net and when.
"""

_REFS = r"""
- Goodfellow, Bengio & Courville — *Deep Learning*, ch. 7 (regularization for deep learning).
- Hastie, Tibshirani & Friedman — *ESL*, §3.4 (ridge/lasso) & ch. 7 (model assessment).
- Srivastava et al. (2014) — *Dropout*.
- In this lab: ML **M0** (over/underfitting), **M6** (validation & model selection),
  the **Deep-nets** page (trigger the overfitting this page cures).
"""


st.title("Regularization — taming overfitting")
st.caption("Dial L2 weight decay up and down on a noisy dataset: watch the boundary go from "
           "wiggly to smooth and the train/test gap close at the sweet spot.")

tab_tune, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎚 Tune it", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_tune:
    st.markdown("A high-capacity MLP `(40, 40)` on **noisy two-moons**. Move the "
                "regularization strength **λ** and watch the boundary and the train/test gap:")
    cc = st.columns(2)
    alpha = cc[0].select_slider("λ  (L2 strength, sklearn alpha)", ALPHAS, value=1e-4,
                                format_func=lambda a: f"{a:g}", key="reg_a")
    seed = cc[1].slider("seed", 0, 30, 0, key="reg_seed")

    r = fit_alpha(alpha, int(seed))
    m = st.columns(3)
    m[0].metric("train accuracy", f"{r['train']:.0%}")
    m[1].metric("test accuracy", f"{r['test']:.0%}")
    m[2].metric("train − test gap", f"{(r['train']-r['test']):.0%}",
                help="big gap = overfitting")

    left, right = st.columns(2)
    with left:
        st.caption("Decision boundary at this λ")
        fig, ax = plt.subplots(figsize=(4.4, 3.9))
        xx, yy = np.meshgrid(np.array(r["gx"]), np.array(r["gy"]))
        ax.contourf(xx, yy, np.array(r["Z"]), levels=20, cmap="RdBu", alpha=0.7, vmin=0, vmax=1)
        ax.contour(xx, yy, np.array(r["Z"]), [0.5], colors="k", linewidths=1.3)
        cmap = ListedColormap(["#A32D2D", "#185FA5"])
        Xtr, ytr = np.array(r["Xtr"]), np.array(r["ytr"])
        Xte, yte = np.array(r["Xte"]), np.array(r["yte"])
        ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap=cmap, s=16, edgecolors="k", linewidths=0.3)
        ax.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap=cmap, s=40, marker="^",
                   edgecolors="k", linewidths=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("● train  ▲ test", fontsize=10)
        st.pyplot(fig, width="stretch")
    with right:
        st.caption("Accuracy vs λ — the sweet spot")
        tr, te = sweep(int(seed))
        fig2, ax2 = plt.subplots(figsize=(4.4, 3.9))
        ax2.semilogx(ALPHAS, tr, "-o", color="#185FA5", label="train")
        ax2.semilogx(ALPHAS, te, "-o", color="#A32D2D", label="test")
        best = int(np.argmax(te))
        ax2.axvline(ALPHAS[best], color="#1D9E75", ls="--", lw=1.3)
        ax2.scatter([ALPHAS[best]], [te[best]], color="#1D9E75", zorder=5, s=60)
        ax2.axvline(alpha, color="#9C9B95", ls=":", lw=1.2)
        ax2.set_xlabel("λ (log)"); ax2.set_ylabel("accuracy")
        ax2.legend(loc="lower left", fontsize=8)
        ax2.set_title(f"best test λ ≈ {ALPHAS[best]:g}", fontsize=10)
        st.pyplot(fig2, width="stretch")

    st.info("L2 makes large weights costly → smaller weights → a smoother boundary. Too "
            "little λ overfits (left of the green line), too much underfits (right). The "
            "green dashed line is the validation-best λ; grey dotted is your current pick.",
            icon=":material/tune:")

with tab_theory:
    st.markdown(_THEORY, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="reg")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
