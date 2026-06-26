import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

import lab

st.title("Tests")
st.caption("`pytest -q` — the safety net that proves core/ is correct as it grows.")

if st.button("Run pytest", icon=":material/play_arrow:", type="primary"):
    with st.spinner("Running pytest…"):
        rc, out = lab.run_pytest()

    passed = int((re.search(r"(\d+) passed", out) or [0, 0])[1])
    skipped = int((re.search(r"(\d+) skipped", out) or [0, 0])[1])
    failed = int((re.search(r"(\d+) failed", out) or [0, 0])[1])

    c1, c2, c3 = st.columns(3)
    c1.metric("Passed", passed)
    c2.metric("Skipped", skipped)
    c3.metric("Failed", failed)

    (st.success if rc == 0 else st.error)(f"exit code {rc}")
    st.code(out, language="text")
else:
    st.info("Skipped grad-check tests turn green once you implement `core/engine.py` (e04).")
