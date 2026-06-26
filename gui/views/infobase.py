import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

import lab

st.title("Infobase")

docs = lab.list_infobase()
if not docs:
    st.info("No notes yet. Write some in `infobase/`.")
    st.stop()


def label(p: pathlib.Path) -> str:
    rel = p.relative_to(lab.INFOBASE)
    return str(rel)


choice = st.sidebar.radio("Notes", docs, format_func=label)
st.caption(f"`infobase/{label(choice)}`")
st.markdown(lab.read(choice))
