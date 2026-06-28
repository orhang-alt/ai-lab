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

import streamlit as st

# Make the lab root importable so `import core` works without `pip install -e .`
# (e.g. on Streamlit Community Cloud, which only installs requirements.txt).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="AI Lab", page_icon="🧠", layout="wide")

# Shared across modules — a Python scratchpad with numpy + the lab's core preloaded.
SANDBOX = st.Page("views/sandbox.py", title="Sandbox", icon=":material/code:")

ANN_PAGES = [
    st.Page("views/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page("views/playground.py", title="Neuron playground", icon=":material/tune:"),
    st.Page("views/two_neurons.py", title="Two neurons", icon=":material/hub:"),
    st.Page("views/backprop.py", title="Backprop", icon=":material/sync_alt:"),
    st.Page("views/mlp.py", title="MLP (train it)", icon=":material/network_node:"),
    st.Page("views/optimizers.py", title="Optimizers", icon=":material/trending_down:"),
    st.Page("views/neurons_compute.py", title="Neurons → computer", icon=":material/calculate:"),
    st.Page("views/the_chain.py", title="The big picture", icon=":material/route:"),
    st.Page("views/attention.py", title="Attention (LLMs)", icon=":material/auto_awesome:"),
    st.Page("views/experiments.py", title="Experiments", icon=":material/science:"),
    st.Page("views/infobase.py", title="Infobase", icon=":material/menu_book:"),
    st.Page("views/tests.py", title="Tests", icon=":material/check_circle:"),
    SANDBOX,
]
ML_PAGES = [
    st.Page("views/ml_overview.py", title="ML overview", icon=":material/dashboard:", default=True),
    st.Page("views/ml_foundations.py", title="M0 · Foundations", icon=":material/school:"),
    st.Page("views/ml_regression.py", title="M1 · Regression", icon=":material/timeline:"),
    st.Page("views/ml_classification.py", title="M2 · Classification", icon=":material/scatter_plot:"),
    st.Page("views/ml_trees.py", title="M3 · Trees & ensembles", icon=":material/account_tree:"),
    st.Page("views/ml_svm.py", title="M4 · SVM & kernels", icon=":material/linear_scale:"),
    st.Page("views/ml_unsupervised.py", title="M5 · Unsupervised", icon=":material/bubble_chart:"),
    st.Page("views/ml_model_selection.py", title="M6 · Model selection", icon=":material/checklist:"),
    st.Page("views/ml_practical.py", title="M7 · Practical ML", icon=":material/build:"),
    st.Page("views/ml_python.py", title="M8 · ML in Python", icon=":material/code:"),
    SANDBOX,
]
MATH_PAGES = [
    st.Page("views/math_overview.py", title="Math overview", icon=":material/dashboard:", default=True),
    st.Page("views/math_vectors.py", title="X1 · Vectors & dot product", icon=":material/call_made:"),
    st.Page("views/math_calculus.py", title="X2 · Calculus & gradients", icon=":material/show_chart:"),
    st.Page("views/math_probability.py", title="X3 · Probability & stats", icon=":material/bar_chart:"),
    st.Page("views/math_optimization.py", title="X4 · Optimization", icon=":material/trending_down:"),
    st.Page("views/math_information.py", title="X5 · Information theory", icon=":material/data_usage:"),
    st.Page("views/math_numerical.py", title="X6 · Numerical computing", icon=":material/calculate:"),
    SANDBOX,
]
MODULES = {"ANN": ANN_PAGES, "ML": ML_PAGES, "Math": MATH_PAGES}

# The Sandbox runs arbitrary Python in-process, so expose it ONLY when explicitly
# enabled — start.sh sets AILAB_ENABLE_SANDBOX=1 locally; it's unset on a public
# deploy (Streamlit Cloud), so the Sandbox is hidden there.
SANDBOX_ENABLED = os.environ.get("AILAB_ENABLE_SANDBOX") == "1"
if not SANDBOX_ENABLED:
    for _pages in MODULES.values():
        if SANDBOX in _pages:
            _pages.remove(SANDBOX)

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
