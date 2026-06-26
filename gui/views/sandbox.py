import contextlib
import io
import os
import pathlib
import subprocess
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

LAB_ROOT = pathlib.Path(__file__).resolve().parents[2]  # ai-lab/

_NEURON = '''import numpy as np
from core.neuron import Neuron

# A single neuron as an AND gate (step activation). Edit weights/bias to try OR, NAND...
gate = Neuron(2, activation="step", weights=[1, 1], bias=-1.5)
for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    print(x, "->", int(gate.forward(x)))
'''

_PERCEPTRON = '''import numpy as np
from core.neuron import Neuron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
y = np.array([0, 0, 0, 1])              # AND target
n = Neuron(2, activation="step", weights=[0.0, 0.0], bias=0.0)
lr = 0.1
for epoch in range(10):
    errors = 0
    for xi, yi in zip(X, y):
        pred = int(n.forward(xi))
        update = lr * (yi - pred)        # perceptron rule
        n.w += update * xi
        n.b += update
        errors += int(yi != pred)
    print(f"epoch {epoch}: errors={errors}")
    if errors == 0:
        break
print("learned:", n)
'''

EXAMPLES = {
    "NumPy basics": '''import numpy as np
a = np.array([1, 2, 3]); b = np.array([4, 5, 6])
print("a + b =", a + b)
print("dot   =", a @ b, " (= |a||b|cosθ)")
print("mean  =", a.mean(), "| L2 norm =", np.linalg.norm(a))
M = np.arange(6).reshape(2, 3)
print("M =\\n", M, "\\nM @ b =", M @ b)
''',
    "Single neuron (ANN e01)": _NEURON,
    "Perceptron rule (ANN e02)": _PERCEPTRON,
    "Least squares / OLS (ML M1)": '''import numpy as np
rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, 50)
y = 1.5 * x + 0.5 + rng.normal(0, 1, 50)      # true slope 1.5, intercept 0.5
X = np.c_[np.ones_like(x), x]
b_hat, w_hat = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"fitted slope = {w_hat:.3f}  (true 1.5)")
print(f"fitted intercept = {b_hat:.3f}  (true 0.5)")
''',
    "Plot activations (inline)": '''import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-6, 6, 300)
plt.plot(x, 1 / (1 + np.exp(-x)), label="sigmoid")
plt.plot(x, np.tanh(x), label="tanh")
plt.plot(x, np.maximum(0, x), label="relu")
plt.legend(); plt.title("activation functions"); plt.grid(alpha=.3)
''',
}

st.title("🐍 Sandbox")
st.caption("A scratchpad with numpy and the lab's `core` preloaded. Test exercises, experiment, "
           "and plot — without leaving the app. Variables persist across runs (like a notebook).")

st.session_state.setdefault("sandbox_code", _NEURON)


def _load_example():
    name = st.session_state.get("sandbox_example_sel")
    if name in EXAMPLES:
        st.session_state["sandbox_code"] = EXAMPLES[name]


def _run_subprocess(code, timeout):
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUTF8": "1"}
    try:
        p = subprocess.run([sys.executable, "-c", code], cwd=str(LAB_ROOT),
                           capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return 124, f"(killed: exceeded {timeout}s timeout)"


c = st.columns([0.42, 0.16, 0.22, 0.20])
c[0].selectbox("example", ["—", *EXAMPLES], key="sandbox_example_sel", label_visibility="collapsed")
c[1].button("Load", on_click=_load_example)
safe = c[2].toggle("Safe run", key="sandbox_safe",
                   help="Run in a separate process with a timeout. Use for code that might loop "
                        "forever. Text output only (no inline plots, no persistent variables).")
timeout = c[3].number_input("timeout (s)", 2, 120, 10, key="sandbox_timeout", disabled=not safe)

st.text_area("code", key="sandbox_code", height=320, label_visibility="collapsed")

r = st.columns([0.18, 0.32, 0.50])
run = r[0].button("Run", type="primary", icon=":material/play_arrow:")
if r[1].button("Reset namespace", icon=":material/restart_alt:"):
    st.session_state.pop("sandbox_ns", None)
    st.toast("Namespace cleared.")

if run:
    code = st.session_state.get("sandbox_code", "")
    st.markdown("**Output**")
    if safe:
        rc, out = _run_subprocess(code, int(timeout))
        st.code(out or "(no output)", language="text")
        (st.caption if rc == 0 else st.error)(f"exit code {rc}")
    else:
        plt.close("all")
        ns = st.session_state.get("sandbox_ns")
        if ns is None:
            ns = {"np": np}
            st.session_state["sandbox_ns"] = ns
        buf, err = io.StringIO(), None
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(code, "<sandbox>", "exec"), ns)
        except Exception:
            err = traceback.format_exc()
        out = buf.getvalue()
        fignums = plt.get_fignums()
        if out:
            st.code(out, language="text")
        for num in fignums:
            st.pyplot(plt.figure(num))
        if err:
            st.error("Exception")
            st.code(err, language="text")
        if not out and not fignums and not err:
            st.caption("(ran fine — no output. Use print(...) or plt.plot(...) to see something.)")

with st.expander("How it works / tips"):
    st.markdown(
        "- **numpy** is preloaded as `np`; the lab's building blocks import normally, e.g. "
        "`from core.neuron import Neuron`.\n"
        "- **Variables persist** across runs (notebook-style). Use **Reset namespace** to start clean.\n"
        "- **Plots** appear inline — just create a matplotlib figure (no need for `plt.show()`).\n"
        "- **Safe run** executes in a separate process with a timeout (no inline plots / no persistence) "
        "— use it if you're worried about an infinite loop freezing the app.\n"
        "- Runs in the lab's venv from the `ai-lab/` directory, so relative paths and `pytest` work."
    )
