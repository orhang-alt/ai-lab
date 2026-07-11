import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import math_lessons2

LESSON = math_lessons2.CALCULUS
FUNCS = {
    "x²": (lambda x: x ** 2, lambda x: 2 * x, "2x"),
    "sin(x)": (np.sin, np.cos, "cos(x)"),
    "eˣ": (np.exp, np.exp, "eˣ"),
    "x³ − 3x": (lambda x: x ** 3 - 3 * x, lambda x: 3 * x ** 2 - 3, "3x² − 3"),
}

st.title("X2 · Calculus & gradients — playground & lesson")
st.caption("The derivative is the slope of the tangent — the limit of secant slopes as h→0. "
           "Shrink h and watch the secant become the tangent.")

lessons.predict(
    'Shrink the secant gap **h** toward 0. What does the secant slope approach — and how is that the same thing backprop relies on?',
    'It approaches the **tangent slope**, i.e. the analytic derivative f′(x); the error → 0 as h → 0. That limit *is* the derivative. Backprop is just the **chain rule** applied to these derivatives, composed layer by layer.',
)

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        name = st.selectbox("function f(x)", list(FUNCS))
        x0 = st.slider("point x", -3.0, 3.0, 1.0, 0.1, key="c_x0")
        h = st.slider("secant gap h", 0.01, 2.0, 1.0, 0.01, key="c_h")
    f, df, dexpr = FUNCS[name]
    m_an = float(df(x0))
    m_sec = float((f(x0 + h) - f(x0)) / h)

    with right:
        xs = np.linspace(-3.6, 3.6, 400)
        ys = f(xs)
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        ax.plot(xs, ys, color="#185FA5", lw=2, label="f(x)")
        tx = np.array([x0 - 1.7, x0 + 1.7])
        ax.plot(tx, f(x0) + m_an * (tx - x0), color="#1D9E75", lw=1.6, label="tangent (true slope)")
        ax.plot([x0, x0 + h], [f(x0), f(x0 + h)], "o--", color="#C0507A", lw=1.6,
                label="secant (slope ≈ f′)")
        ax.axhline(0, color="0.85", lw=0.6)
        ax.set_ylim(float(ys.min()) - 1, float(ys.max()) + 1)
        ax.set_title("derivative = slope of the tangent")
        ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("f′(x) analytic", f"{m_an:.3f}")
    c2.metric("secant slope", f"{m_sec:.3f}")
    c3.metric("abs error", f"{abs(m_sec - m_an):.3f}")
    st.latex(rf"f'(x) = {dexpr} \quad\Rightarrow\quad f'({x0:.1f}) = {m_an:.3f}")
    st.info("As **h → 0** the secant slope converges to the analytic derivative (the error "
            "shrinks). That limit *is* the derivative — and the chain rule on it is backprop.",
            icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mathcalc")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** $\frac{d}{dx}3x^4=12x^3$; $\frac{d}{dx}\sin x=\cos x$; $\frac{d}{dx}e^{2x}=2e^{2x}$; $\frac{d}{dx}(5x^2+2x-1)=10x+2$.

**2.** $\frac{d}{dx}(3x+1)^5=5(3x+1)^4\cdot3=15(3x+1)^4$; $\frac{d}{dx}e^{-x^2}=-2x\,e^{-x^2}$.

**3.** $\partial f/\partial x=2xy$, $\partial f/\partial y=x^2+3y^2$. At $(1,2)$: $\nabla f=(4,\,13)$.

**4.** $\nabla(x^2+y^2)=(2x,2y)=2(x,y)$ — at every point it points **radially outward** from the origin (steepest ascent of the bowl).""",
        label="Pencil & paper 1–4",
    )
    lessons.solution(
        r"""**5–6.** The central difference $\frac{f(x+h)-f(x-h)}{2h}$ matches $\frac{d}{dx}x^2=2x$ (the e06 gradient check); the tangent to $x^2$ at $x=2$ has slope 4.

**7.** Backprop *is* the chain rule: for a 2-layer net, $\partial L/\partial w^{(1)}$ is the product of local derivatives from the loss back through layer 2's activation and weights into layer 1 (see the **Backprop** page).""",
        label="Code & bridge 5–7",
    )
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
