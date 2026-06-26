import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

import lab

st.title("Experiments")

tiers = lab.list_tiers()
options = [(t, e) for t, exps in tiers.items() for e in exps]
labels = [f"{lab.tier_label(t)}  —  {e.label}" for t, e in options]

idx = st.selectbox("Experiment", range(len(options)), format_func=lambda i: labels[i])
tier, exp = options[idx]

overview, code, run, notes = st.tabs(["📄 Overview", "💻 Code", "▶ Run", "📝 Notes"])

with overview:
    st.markdown(lab.read(exp.readme) or "_No README.md_")

with code:
    st.code(lab.read(exp.run_py) or "# empty", language="python")
    st.caption(f"`{exp.run_py.relative_to(lab.LAB_ROOT)}`")

with run:
    st.caption("Runs `run.py` in the venv (headless). Plots open via the CLI `--plot` flag.")
    if st.button("Run experiment", icon=":material/play_arrow:", type="primary"):
        with st.spinner("Running…"):
            rc, out = lab.run_script(exp.run_py)
        (st.success if rc == 0 else st.warning)(f"exit code {rc}")
        st.code(out or "(no output)", language="text")

with notes:
    current = lab.read(exp.notes)
    edited = st.text_area("notes.md", current, height=320, label_visibility="collapsed")
    if st.button("Save notes", icon=":material/save:"):
        exp.notes.write_text(edited)
        st.success("Saved.")
