import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

import i18n
import lab

LANG = i18n.lang()
COPY = {
    "en": {
        "title": "Experiments",
        "tier": "Tier",
        "experiment": "Experiment",
        "tabs": ["Overview", "Code", "Run"],
        "no_readme": "_No README.md_",
        "run_caption": "Runs `run.py` in the venv (headless). Plots open via the CLI `--plot` flag.",
        "run_button": "Run experiment",
        "running": "Running...",
        "exit_code": "exit code",
        "no_output": "(no output)",
    },
    "tr": {
        "title": "Deneyler",
        "tier": "Seviye",
        "experiment": "Deney",
        "tabs": ["Genel Bakış", "Kod", "Çalıştır"],
        "no_readme": "_README.md yok_",
        "run_caption": "`run.py` dosyasını venv içinde çalıştırır (headless). Grafikler CLI `--plot` bayrağıyla açılır.",
        "run_button": "Deneyi çalıştır",
        "running": "Çalıştırılıyor...",
        "exit_code": "çıkış kodu",
        "no_output": "(çıktı yok)",
    },
}
C = COPY.get(LANG, COPY["en"])

st.title(C["title"])

tiers = lab.list_tiers()
tier_names = list(tiers.keys())

# --- visible selectors, right in the main area -----------------------------
tsel = st.segmented_control(
    C["tier"], tier_names, format_func=lab.tier_label, default=tier_names[0], key="exp_tier"
) or tier_names[0]

exps = tiers[tsel]
e_idx = st.segmented_control(
    C["experiment"], list(range(len(exps))),
    format_func=lambda i: f"{exps[i].id} · {exps[i].name}",
    default=0, key=f"exp_idx_{tsel}",
)
if e_idx is None:
    e_idx = 0
exp = exps[e_idx]

st.caption(f"{lab.tier_label(tsel)}  ·  `{exp.run_py.relative_to(lab.LAB_ROOT)}`")
st.divider()

overview, code, run = st.tabs([f"📄 {C['tabs'][0]}", f"💻 {C['tabs'][1]}", f"▶ {C['tabs'][2]}"])

with overview:
    st.markdown(lab.read_localized(exp.readme, LANG) or C["no_readme"])

with code:
    st.code(lab.read(exp.run_py) or "# empty", language="python")
    st.caption(f"`{exp.run_py.relative_to(lab.LAB_ROOT)}`")

with run:
    st.caption(C["run_caption"])
    if st.button(C["run_button"], icon=":material/play_arrow:", type="primary"):
        with st.spinner(C["running"]):
            rc, out = lab.run_script(exp.run_py)
        (st.success if rc == 0 else st.warning)(f"{C['exit_code']} {rc}")
        st.code(out or C["no_output"], language="text")
