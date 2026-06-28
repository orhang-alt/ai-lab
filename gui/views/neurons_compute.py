"""Neurons → a computer (ANN module).

A single neuron is a logic gate; wire gates together and neurons do *arithmetic* — and,
since NAND is functionally complete, in principle any computation. Here we build a half
adder and a full adder entirely out of TLU neurons (core.neuron, step activation) and
watch them add binary numbers.
"""

import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import pandas as pd
import streamlit as st

from core.neuron import Neuron

# Each gate is one TLU neuron:  y = step(w·x − θ)
_AND = Neuron(2, "step", [1, 1], -1.5)
_OR = Neuron(2, "step", [1, 1], -0.5)
_XOR_OUT = Neuron(3, "step", [1, 1, -2], -0.5)   # takes [a, b, AND(a,b)]


def AND(a, b):
    return int(_AND.forward([a, b]))


def OR(a, b):
    return int(_OR.forward([a, b]))


def XOR(a, b):
    return int(_XOR_OUT.forward([a, b, AND(a, b)]))   # needs 2 neurons (hidden AND + output)


def half_adder(a, b):
    return XOR(a, b), AND(a, b)                        # (sum, carry)


def full_adder(a, b, cin):
    s1, c1 = half_adder(a, b)
    s2, c2 = half_adder(s1, cin)
    return s2, OR(c1, c2)                              # (sum, carry-out)


_HALF_ADDER_SVG = '''<div style="text-align:center;margin:0.4rem 0"><svg viewBox="0 0 560 230" style="width:100%;max-width:560px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A half adder built from neurons: an AND neuron makes the Carry, a second neuron makes the Sum (XOR)."><defs><marker id="haah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#33312E"/></marker></defs><rect x="1" y="1" width="558" height="228" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><text x="38" y="73" font-family="sans-serif" font-size="16" fill="#33312E">A</text><text x="38" y="173" font-family="sans-serif" font-size="16" fill="#33312E">B</text><g stroke="#33312E" stroke-width="1.7" fill="none"><line x1="56" y1="68" x2="214" y2="78" marker-end="url(#haah)"/><line x1="56" y1="68" x2="214" y2="168" marker-end="url(#haah)"/><line x1="56" y1="168" x2="214" y2="92" marker-end="url(#haah)"/><line x1="56" y1="168" x2="214" y2="182" marker-end="url(#haah)"/></g><circle cx="250" cy="80" r="34" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="2"/><text x="250" y="85" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#5A3E14">AND</text><circle cx="250" cy="180" r="34" fill="#CFE3F5" stroke="#5B8FC2" stroke-width="2"/><text x="250" y="185" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#0C447C">XOR</text><line x1="250" y1="114" x2="250" y2="146" stroke="#33312E" stroke-width="1.7" marker-end="url(#haah)"/><line x1="284" y1="80" x2="468" y2="80" stroke="#33312E" stroke-width="1.7" marker-end="url(#haah)"/><line x1="284" y1="180" x2="468" y2="180" stroke="#33312E" stroke-width="1.7" marker-end="url(#haah)"/><text x="476" y="85" font-family="sans-serif" font-size="15" fill="#9A6A2A">Carry</text><text x="476" y="185" font-family="sans-serif" font-size="15" fill="#0C447C">Sum</text><text x="280" y="218" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#9C9B95">Carry = AND(A,B)   ·   Sum = XOR(A,B)</text></svg></div>'''

st.title("Neurons → a computer")
st.caption("A single neuron is a logic gate. Wire gates together and neurons do arithmetic — "
           "and, since NAND alone is universal, in principle ANY computation.")

st.markdown(
    "**Step 1 — a neuron is a gate.** Each of these is *one* TLU neuron "
    "`y = step(w·x − θ)`:\n\n"
    "| gate | weights w | threshold θ |\n|---|---|---|\n"
    "| AND | [1, 1] | 1.5 |\n| OR | [1, 1] | 0.5 |\n| NAND | [−1, −1] | −1.5 |\n"
    "| NOT | [−1] | −0.5 |\n\n"
    "XOR isn't linearly separable, so it takes **two** neurons (a hidden AND + an output) "
    "— exactly the *Two neurons* page."
)

st.divider()
st.subheader("Step 2 — wire gates into an adder (neurons doing arithmetic)")
st.markdown(
    "To **add two bits** you need two answers: the **Sum** bit and the **Carry** bit. "
    "Look at `1 + 1`: the answer is `2`, which in binary is `10` — Sum = 0, Carry = 1. "
    "That's exactly **Sum = XOR(A,B)** and **Carry = AND(A,B)** — two neurons, the *half adder* below."
)
st.markdown(_HALF_ADDER_SVG, unsafe_allow_html=True)
mode = st.segmented_control(
    "Circuit", ["Half adder (A + B)", "Full adder (A + B + Cin)"],
    default="Half adder (A + B)", key="nc_mode",
) or "Half adder (A + B)"

if mode.startswith("Half"):
    c1, c2 = st.columns(2)
    a = int(c1.checkbox("A = 1", key="ha_a"))
    b = int(c2.checkbox("B = 1", key="ha_b"))
    s, carry = half_adder(a, b)
    m = st.columns(3)
    m[0].metric("A + B (decimal)", a + b)
    m[1].metric("Carry", carry)
    m[2].metric("Sum", s)
    st.latex(rf"{a} + {b} \;=\; 2\cdot{carry} + {s} \;=\; {2 * carry + s}")
    rows = [{"A": x, "B": y, "Carry": half_adder(x, y)[1], "Sum": half_adder(x, y)[0]}
            for x, y in itertools.product([0, 1], repeat=2)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="content")
    st.info("Built from **2 neurons**: Carry = AND(A,B), Sum = XOR(A,B). Tick both A and B: "
            "Carry=1, Sum=0 — that's binary **10 = 2**. The neurons added 1+1.",
            icon=":material/calculate:")
else:
    c1, c2, c3 = st.columns(3)
    a = int(c1.checkbox("A = 1", key="fa_a"))
    b = int(c2.checkbox("B = 1", key="fa_b"))
    cin = int(c3.checkbox("Cin = 1", key="fa_cin"))
    s, cout = full_adder(a, b, cin)
    m = st.columns(3)
    m[0].metric("A + B + Cin", a + b + cin)
    m[1].metric("Carry-out", cout)
    m[2].metric("Sum", s)
    st.latex(rf"{a} + {b} + {cin} \;=\; 2\cdot{cout} + {s} \;=\; {2 * cout + s}")
    rows = [{"A": x, "B": y, "Cin": z, "Cout": full_adder(x, y, z)[1], "Sum": full_adder(x, y, z)[0]}
            for x, y, z in itertools.product([0, 1], repeat=3)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="content")
    st.info("A full adder = two half-adders + an OR. Chain n of these and neurons add n-bit "
            "numbers — the core of a CPU's arithmetic unit.", icon=":material/calculate:")

st.divider()
st.subheader("Step 3 — why this means 'any computation'")
st.markdown(
    "- A neuron can be a **NAND** gate, and **NAND is functionally complete** — every logic "
    "circuit can be built from NANDs alone.\n"
    "- So networks of neurons can compute **any boolean function**, and (with memory / feedback) "
    "anything a computer can.\n"
    "- A real **trained** network does the same thing with **soft, learned gates** (sigmoids "
    "instead of hard steps) and weights found by **gradient descent** instead of hand-set — "
    "that's the only real difference between this adder and a deep net.\n"
    "- And physically it's all one operation, repeated: **weighted sums (matrix multiplies) + "
    "a nonlinearity**."
)

st.divider()
st.subheader("Logic circuits, in 60 seconds")
st.markdown(
    "The very same *neuron = gate* idea scales all the way up to a real computer. Five rungs "
    "on the ladder:\n\n"
    "1. **Gates** — AND, OR, NOT, NAND. Each is one neuron (XOR needs two). The alphabet.\n"
    "2. **Combinational logic** — wire gates together with **no loops** so outputs depend only "
    "on the current inputs: **adders** (above), **multiplexers** (pick one of several inputs), "
    "**decoders**, comparators. This is *arithmetic and choosing*.\n"
    "3. **Memory** — now add a **feedback loop**: route an output back into the input, and a "
    "pair of gates can **hold a bit** (a *latch* / flip-flop). Loops give you **state** — "
    "something a plain feed-forward network simply does not have.\n"
    "4. **Sequential logic** — gates + memory + a **clock** that ticks. The machine now has "
    "*steps*: registers, counters, state machines — it can follow a procedure.\n"
    "5. **A CPU** is just a vast pile of exactly these — gates for the arithmetic unit, latches "
    "for registers and RAM, and control logic sequencing it all. Same Lego bricks, billions of them.\n\n"
    "So the whole arc is: **neuron → gate → circuit → (＋ feedback) memory → (＋ a clock) a "
    "programmable computer.** The jump that matters is the **feedback loop** — that's what turns "
    "a calculator into a computer."
)

st.divider()
st.subheader("Is an ANN model really a computer?")
st.markdown(
    "Short answer: a standard **feed-forward** neural network is **not** a general-purpose "
    "computer — it's a fixed **function**. But the line is subtle and worth drawing carefully."
)
with st.expander("What a plain MLP actually is"):
    st.markdown(
        "Once trained, a feed-forward net maps **inputs → outputs in a fixed number of steps**, "
        "with **no loops** and **no writable memory**. It *can* compute any boolean function "
        "(we just built arithmetic) and approximate any continuous function — so it is "
        "**universal as a function**. But it cannot 'loop until done', cannot store and recall "
        "arbitrary data, cannot take an unbounded number of steps. So an MLP is much more like "
        "**one big combinational circuit** than like a computer."
    )
with st.expander("What makes something a computer"):
    st.markdown(
        "Three ingredients a plain MLP lacks:\n"
        "- **Memory** — writable, readable state.\n"
        "- **Control flow** — loops/branches that run a *variable* number of steps.\n"
        "- **Programmability** — behaviour you can change without rebuilding the hardware.\n\n"
        "Crucially, these come from **feedback/recurrence**, *not* from depth. Stacking more "
        "layers makes a richer function; it does **not** by itself make a computer."
    )
with st.expander("Where neural nets cross the line"):
    st.markdown(
        "- **Recurrent nets (RNN / LSTM)** add feedback = memory over time. In theory, an RNN "
        "with enough precision is **Turing-complete** — it *can* express general computation.\n"
        "- **Transformers / LLMs** are feed-forward per step, but run **autoregressively** "
        "(feed each output back in as input) and can use a **scratchpad** ('chain of thought') "
        "and external **tools**. That effectively buys them loops and memory — which is why an "
        "LLM can carry out genuine multi-step procedures."
    )
st.markdown(
    "**The honest takeaway.** A neuron is a (soft) **logic gate**; a network is a **circuit**; "
    "a *trained* network is a circuit whose gates were **learned from data** instead of wired by "
    "an engineer. That makes it a **universal function approximator** — and, *with feedback and "
    "memory*, a genuine computer. The deep difference from your laptop's CPU isn't the "
    "ingredients (it's all gates and memory either way) — it's that the network's 'program' is "
    "**learned weights**: fuzzy, statistical, and robust, rather than exact hand-written instructions."
)
st.info("Rule of thumb: **depth → a better function; feedback → a computer.** Our adder is a "
        "circuit; an LLM borrows loops and memory to act like a computer.", icon=":material/memory:")
