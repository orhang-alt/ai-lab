"""AI Lab GUI — entrypoint.

Three modules, selectable in the sidebar:
  - ANN  : neural networks, single neuron -> GPT (what you've been using).
  - ML   : classical machine learning, same approach (theory + playground + quiz).
  - Math : the mathematical basics needed to study ML & ANN.

Launch:
    cd ai-lab && source .venv/bin/activate
    streamlit run gui/app.py
"""

import os
import pathlib
import sys

# Matplotlib must use a non-GUI backend: Streamlit runs scripts on a worker thread, and the
# macOS GUI backend refuses to create figures off the main thread. Set before any view loads.
os.environ.setdefault("MPLBACKEND", "Agg")

import streamlit as st

# Make the lab root importable so `import core` works without `pip install -e .`
# (e.g. on Streamlit Community Cloud, which only installs requirements.txt).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="AI Lab", page_icon="🧠", layout="wide")

# Shared across modules — a Python scratchpad with numpy + the lab's core preloaded.
SANDBOX = st.Page("views/sandbox.py", title="Sandbox", icon=":material/code:")

# Pages are grouped into sections that follow the learning path. Passing a {section: [pages]}
# dict to st.navigation renders the keys as sidebar section headers (much less crowded).
ANN = {
    "Overview": [
        st.Page("views/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
        st.Page("views/the_chain.py", title="The big picture", icon=":material/route:"),
    ],
    "Level 1 · The neuron": [
        st.Page("views/playground.py", title="Neuron playground", icon=":material/tune:"),
        st.Page("views/two_neurons.py", title="Two neurons", icon=":material/hub:"),
        st.Page("views/neurons_compute.py", title="Neurons → computer", icon=":material/calculate:"),
    ],
    "Level 2 · Training a net": [
        st.Page("views/backprop.py", title="Backprop", icon=":material/sync_alt:"),
        st.Page("views/mlp.py", title="MLP (train it)", icon=":material/network_node:"),
        st.Page("views/optimizers.py", title="Optimizers", icon=":material/trending_down:"),
        st.Page("views/deep_playground.py", title="Deep nets (2D)", icon=":material/blur_on:"),
        st.Page("views/regularization.py", title="Regularization", icon=":material/tune:"),
    ],
    "Level 3 · Architectures": [
        st.Page("views/cnn.py", title="CNN (images)", icon=":material/image:"),
        st.Page("views/rnn.py", title="RNN (sequences)", icon=":material/repeat:"),
    ],
    "Level 4 · Attention & Transformers": [
        st.Page("views/tokenization.py", title="Tokenization", icon=":material/content_cut:"),
        st.Page("views/attention.py", title="Attention (LLMs)", icon=":material/auto_awesome:"),
        st.Page("views/transformer.py", title="Tiny GPT", icon=":material/smart_toy:"),
    ],
    "Level 5 · LLMs in practice": [
        st.Page("views/sampling.py", title="Decoding (sampling)", icon=":material/casino:"),
        st.Page("views/embeddings.py", title="Embeddings & RAG", icon=":material/database:"),
        st.Page("views/posttraining.py", title="Post-training (RLHF)", icon=":material/psychology:"),
    ],
    "Lab": [
        st.Page("views/experiments.py", title="Experiments", icon=":material/science:"),
        st.Page("views/infobase.py", title="Infobase", icon=":material/menu_book:"),
        st.Page("views/tests.py", title="Tests", icon=":material/check_circle:"),
    ],
}
ML = {
    "Overview": [
        st.Page("views/ml_overview.py", title="ML overview", icon=":material/dashboard:", default=True),
        st.Page("views/ml_foundations.py", title="M0 · Foundations", icon=":material/school:"),
    ],
    "Supervised (M1–M4)": [
        st.Page("views/ml_regression.py", title="M1 · Regression", icon=":material/timeline:"),
        st.Page("views/ml_classification.py", title="M2 · Classification", icon=":material/scatter_plot:"),
        st.Page("views/ml_trees.py", title="M3 · Trees & ensembles", icon=":material/account_tree:"),
        st.Page("views/ml_svm.py", title="M4 · SVM & kernels", icon=":material/linear_scale:"),
    ],
    "Unsupervised (M5)": [
        st.Page("views/ml_unsupervised.py", title="M5 · Unsupervised", icon=":material/bubble_chart:"),
    ],
    "Practice (M6–M8)": [
        st.Page("views/ml_model_selection.py", title="M6 · Model selection", icon=":material/checklist:"),
        st.Page("views/ml_practical.py", title="M7 · Practical ML", icon=":material/build:"),
        st.Page("views/ml_python.py", title="M8 · ML in Python", icon=":material/code:"),
    ],
}
MATH = {
    "Overview": [
        st.Page("views/math_overview.py", title="Math overview", icon=":material/dashboard:", default=True),
    ],
    "Algebra & calculus": [
        st.Page("views/math_vectors.py", title="X1 · Vectors & dot product", icon=":material/call_made:"),
        st.Page("views/math_calculus.py", title="X2 · Calculus & gradients", icon=":material/show_chart:"),
    ],
    "Probability & optimization": [
        st.Page("views/math_probability.py", title="X3 · Probability & stats", icon=":material/bar_chart:"),
        st.Page("views/math_optimization.py", title="X4 · Optimization", icon=":material/trending_down:"),
    ],
    "Information & numerics": [
        st.Page("views/math_information.py", title="X5 · Information theory", icon=":material/data_usage:"),
        st.Page("views/math_numerical.py", title="X6 · Numerical computing", icon=":material/calculate:"),
    ],
}
MODULES = {"ANN": ANN, "ML": ML, "Math": MATH}

# The Sandbox runs arbitrary Python in-process, so expose it ONLY when explicitly
# enabled — start.sh sets AILAB_ENABLE_SANDBOX=1 locally; it's unset on a public
# deploy (Streamlit Cloud), so the Sandbox is hidden there.
SANDBOX_ENABLED = os.environ.get("AILAB_ENABLE_SANDBOX") == "1"
if SANDBOX_ENABLED:
    for _sections in MODULES.values():
        _sections["Tools"] = [SANDBOX]

# --- module selector (the "select at the beginning") ------------------------
module = st.sidebar.segmented_control(
    "Module", list(MODULES), default="ANN", key="module",
    help="ANN = neural networks (single neuron → GPT). "
         "ML = classical machine learning. "
         "Math = the mathematical basics for both.",
) or "ANN"
st.sidebar.caption(f"Module: **{module}**")
st.sidebar.divider()

nav = st.navigation(MODULES[module])
nav.run()

# --- attribution footer, pinned to the bottom of the sidebar ---------------
st.sidebar.divider()
st.sidebar.caption("**Orhan Gökçöl's** personal AI lab")
st.sidebar.caption("Free to use. 🧠")
