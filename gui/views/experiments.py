import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

import lab

st.title("Experiments")

tiers = lab.list_tiers()
tier_names = list(tiers.keys())

# --- visible selectors, right in the main area -----------------------------
tsel = st.segmented_control(
    "Tier", tier_names, format_func=lab.tier_label, default=tier_names[0], key="exp_tier"
) or tier_names[0]

exps = tiers[tsel]
e_idx = st.segmented_control(
    "Experiment", list(range(len(exps))),
    format_func=lambda i: f"{exps[i].id} · {exps[i].name}",
    default=0, key=f"exp_idx_{tsel}",
)
if e_idx is None:
    e_idx = 0
exp = exps[e_idx]

st.caption(f"{lab.tier_label(tsel)}  ·  `{exp.run_py.relative_to(lab.LAB_ROOT)}`")
st.divider()

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
        try:
            exp.notes.write_text(edited)
            st.success("Saved.")
        except OSError as e:
            st.warning(f"Couldn't save — on a hosted deploy the filesystem is read-only/"
                       f"ephemeral, so notes don't persist. ({e})")
