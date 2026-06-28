import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

import i18n
import lab

LANG = i18n.lang()
COPY = {
    "en": {"title": "Infobase", "empty": "No notes yet. Write some in `infobase/`.", "notes": "Notes"},
    "tr": {"title": "Bilgi tabanı", "empty": "Henüz not yok. `infobase/` içine birkaç not yaz.", "notes": "Notlar"},
}
C = COPY.get(LANG, COPY["en"])

st.title(C["title"])

docs = lab.list_infobase()
if not docs:
    st.info(C["empty"])
    st.stop()


def label(p: pathlib.Path) -> str:
    rel = p.relative_to(lab.INFOBASE)
    return str(rel)


choice = st.sidebar.radio(C["notes"], docs, format_func=label)
localized = lab.localized_path(choice, LANG)
st.caption(f"`infobase/{label(choice)}`")
st.markdown(lab.read(localized))
