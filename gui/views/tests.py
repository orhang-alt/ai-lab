import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

import i18n
import lab

LANG = i18n.lang()
COPY = {
    "en": {
        "title": "Tests",
        "caption": "`pytest -q` — the safety net that proves core/ is correct as it grows.",
        "run": "Run pytest",
        "running": "Running pytest...",
        "passed": "Passed",
        "skipped": "Skipped",
        "failed": "Failed",
        "exit_code": "exit code",
        "info": "Run the test suite to check the neuron, autograd engine, losses, optimizers, and MLP plumbing.",
    },
    "tr": {
        "title": "Testler",
        "caption": "`pytest -q` — core/ büyüdükçe doğruluğu kanıtlayan güvenlik ağı.",
        "run": "Pytest çalıştır",
        "running": "Pytest çalışıyor...",
        "passed": "Geçen",
        "skipped": "Atlanan",
        "failed": "Kalan hata",
        "exit_code": "çıkış kodu",
        "info": "Nöron, autograd motoru, loss fonksiyonları, optimizer'lar ve MLP bağlantısını kontrol etmek için testleri çalıştır.",
    },
}
C = COPY.get(LANG, COPY["en"])

st.title(C["title"])
st.caption(C["caption"])

if st.button(C["run"], icon=":material/play_arrow:", type="primary"):
    with st.spinner(C["running"]):
        rc, out = lab.run_pytest()

    passed = int((re.search(r"(\d+) passed", out) or [0, 0])[1])
    skipped = int((re.search(r"(\d+) skipped", out) or [0, 0])[1])
    failed = int((re.search(r"(\d+) failed", out) or [0, 0])[1])

    c1, c2, c3 = st.columns(3)
    c1.metric(C["passed"], passed)
    c2.metric(C["skipped"], skipped)
    c3.metric(C["failed"], failed)

    (st.success if rc == 0 else st.error)(f"{C['exit_code']} {rc}")
    st.code(out, language="text")
else:
    st.info(C["info"])
