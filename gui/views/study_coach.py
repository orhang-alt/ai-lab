import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

import lab
import ui


st.title("Study Coach")
st.caption("A quiet training loop: run, explain, derive, rebuild from memory.")

ui.hero(
    "Today: make one idea yours",
    "Pick one experiment, run it, fill a checkpoint, answer five questions aloud, then break one thing on purpose.",
)

steps = [
    ("Run", "Run one experiment and copy the most surprising output into notes."),
    ("Explain", "Answer questions from memory before reading the code again."),
    ("Derive", "Rewrite one formula in your own words and name every symbol."),
    ("Break", "Change one assumption so the result fails; record the failure mode."),
]

cols = st.columns(4)
for col, (title, body) in zip(cols, steps):
    with col:
        st.markdown(f"<div class='ailab-step'><strong>{title}</strong><br>{body}</div>", unsafe_allow_html=True)

st.divider()

tab_today, tab_questions, tab_checks, tab_derivations, tab_template = st.tabs(
    ["Today", "Questions", "Self-checks", "Derivations", "Checkpoint"]
)

with tab_today:
    st.subheader("Suggested order")
    st.markdown(
        """
1. `e09_optimizers` — feel what the optimizer changes.
2. `e10_initialization` — see why starting weights matter.
3. `e11_regularization` — separate train loss from generalization.
4. `infobase/04_derivations/adam-update.md` — rewrite Adam from memory.
5. `learning/self_checks/tier2_self_check.md` — diagnose a failed training run.
        """
    )
    st.markdown(
        "<div class='ailab-callout'>Stop when you can explain the failure mode, not when the script passes.</div>",
        unsafe_allow_html=True,
    )

with tab_questions:
    qfiles = {
        "Tier 0": lab.LAB_ROOT / "learning/questions/tier0_neuron.md",
        "Tier 1": lab.LAB_ROOT / "learning/questions/tier1_backprop.md",
        "Tier 2": lab.LAB_ROOT / "learning/questions/tier2_training.md",
    }
    choice = st.segmented_control("Question set", list(qfiles), default="Tier 2")
    st.markdown(lab.read(qfiles[choice]))

with tab_checks:
    cfiles = {
        "Tier 0": lab.LAB_ROOT / "learning/self_checks/tier0_self_check.md",
        "Tier 1": lab.LAB_ROOT / "learning/self_checks/tier1_self_check.md",
        "Tier 2": lab.LAB_ROOT / "learning/self_checks/tier2_self_check.md",
    }
    choice = st.segmented_control("Self-check", list(cfiles), default="Tier 2")
    st.markdown(lab.read(cfiles[choice]))

with tab_derivations:
    derivations = sorted((lab.LAB_ROOT / "infobase/04_derivations").glob("*.md"))
    labels = [p.stem.replace("-", " ") for p in derivations]
    selected = st.selectbox("Derivation", labels, index=0)
    path = derivations[labels.index(selected)]
    st.markdown(lab.read(path))

with tab_template:
    st.markdown(lab.read(lab.LAB_ROOT / "learning/templates/checkpoint.md"))

