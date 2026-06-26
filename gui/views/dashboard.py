import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

import lab

st.title("🧠 AI Lab")
st.caption("From a single neuron to a small GPT — one growing library, many experiments.")


@st.cache_data(show_spinner="Scanning experiments…")
def scan():
    """Run each experiment once; status = done / todo / error."""
    out = {}
    for tier, exps in lab.list_tiers().items():
        out[tier] = [(e, lab.status_of(e)) for e in exps]
    return out


ICON = {"done": "🟢", "todo": "⚪", "error": "🔴"}

top = st.container()
if st.button("Re-scan", icon=":material/refresh:"):
    scan.clear()

data = scan()
flat = [s for rows in data.values() for _, s in rows]
done = flat.count("done")

with top:
    c1, c2, c3 = st.columns(3)
    c1.metric("Experiments", len(flat))
    c2.metric("Done", done)
    c3.metric("Remaining", len(flat) - done)
    st.progress(done / len(flat) if flat else 0.0)

st.divider()

for tier, rows in data.items():
    t_done = sum(1 for _, s in rows if s == "done")
    st.subheader(f"{lab.tier_label(tier)}  ·  {t_done}/{len(rows)}")
    for exp, status in rows:
        cols = st.columns([0.08, 0.5, 0.42])
        cols[0].write(ICON[status])
        cols[1].write(f"**{exp.id}** {exp.name}")
        cols[2].caption(status)

with st.expander("View ROADMAP.md"):
    st.markdown(lab.read(lab.LAB_ROOT / "ROADMAP.md"))
