import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import math_lessons2

LESSON = math_lessons2.PROBABILITY

DISTS = {
    "Uniform [0,1]": lambda rng, shape: rng.uniform(0, 1, shape),
    "Exponential": lambda rng, shape: rng.exponential(1.0, shape),
    "Bernoulli(0.3)": lambda rng, shape: (rng.random(shape) < 0.3).astype(float),
}

USES = {
    "Bernoulli": "Any single yes/no outcome — spam/not, click/no-click, fraud/legit. The output of logistic regression and a sigmoid neuron.",
    "Binomial": "Counts of successes over n independent trials — conversions out of n visitors (A/B tests), defects per batch.",
    "Poisson": "Counts of rare, independent events per interval — arrivals per minute, photons on a sensor, typos per page. Poisson regression models counts.",
    "Geometric": "How many trials until the first success — attempts until a user converts or a packet is delivered (memoryless).",
    "Gaussian": "Noise, regression residuals, standardized features, weight init, the CLT limit, VAE latents — the default when unsure.",
    "Beta": "An unknown probability/rate in [0,1] — click/conversion rates; the prior in Bayesian A/B testing & Thompson-sampling bandits.",
    "Gamma": "Positive, skewed amounts — time-to-failure, claim sizes, rainfall, the wait for k events; conjugate prior for a Poisson rate.",
}


def _explorer():
    left, right = st.columns([0.42, 0.58])
    with left:
        dist = st.selectbox("distribution", list(USES), key="ex_dist")
        if dist == "Bernoulli":
            p = st.slider("p", 0.0, 1.0, 0.5, 0.05, key="ex_bern_p")
            xs, y, mean, var, disc = np.array([0, 1]), np.array([1 - p, p]), p, p * (1 - p), True
        elif dist == "Binomial":
            n = st.slider("n", 1, 40, 10, key="ex_bin_n")
            p = st.slider("p", 0.0, 1.0, 0.5, 0.05, key="ex_bin_p")
            xs = np.arange(0, n + 1)
            y = np.array([math.comb(n, int(k)) * p ** k * (1 - p) ** (n - k) for k in xs])
            mean, var, disc = n * p, n * p * (1 - p), True
        elif dist == "Poisson":
            lam = st.slider("λ (rate)", 0.5, 30.0, 4.0, 0.5, key="ex_pois_lam")
            xs = np.arange(0, int(lam + 4 * np.sqrt(lam) + 5))
            y = np.exp(-lam + xs * np.log(lam) - np.array([math.lgamma(int(k) + 1) for k in xs]))
            mean, var, disc = lam, lam, True
        elif dist == "Geometric":
            p = st.slider("p", 0.05, 1.0, 0.3, 0.05, key="ex_geo_p")
            xs = np.arange(1, 31)
            y, mean, var, disc = (1 - p) ** (xs - 1) * p, 1 / p, (1 - p) / p ** 2, True
        elif dist == "Gaussian":
            mu = st.slider("μ (mean)", -5.0, 5.0, 0.0, 0.5, key="ex_g_mu")
            sigma = st.slider("σ (std)", 0.3, 4.0, 1.0, 0.1, key="ex_g_sig")
            xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
            y = np.exp(-(xs - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
            mean, var, disc = mu, sigma ** 2, False
        elif dist == "Beta":
            a = st.slider("α", 0.5, 6.0, 2.0, 0.5, key="ex_beta_a")
            b = st.slider("β", 0.5, 6.0, 2.0, 0.5, key="ex_beta_b")
            xs = np.linspace(0.001, 0.999, 300)
            B = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
            y = xs ** (a - 1) * (1 - xs) ** (b - 1) / B
            mean, var, disc = a / (a + b), a * b / ((a + b) ** 2 * (a + b + 1)), False
        else:  # Gamma
            a = st.slider("shape α", 0.5, 8.0, 2.0, 0.5, key="ex_gam_a")
            rate = st.slider("rate β", 0.3, 4.0, 1.0, 0.1, key="ex_gam_b")
            xs = np.linspace(0.001, a / rate + 4 * np.sqrt(a) / rate + 1, 300)
            y = np.exp(a * np.log(rate) + (a - 1) * np.log(xs) - rate * xs - math.lgamma(a))
            mean, var, disc = a / rate, a / rate ** 2, False

    with right:
        fig, ax = plt.subplots(figsize=(4.7, 4.0))
        if disc:
            ax.bar(xs, y, color="#185FA5", width=0.4 if dist == "Bernoulli" else 0.85)
            ax.set_ylabel("P(X = k)")
        else:
            ax.plot(xs, y, color="#185FA5", lw=2)
            ax.fill_between(xs, y, alpha=0.25, color="#185FA5")
            ax.set_ylabel("density")
        ax.axvline(mean, color="#C0507A", ls="--", lw=1.5, label=f"mean = {mean:.2f}")
        ax.set_title(f"{dist} distribution")
        ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("mean", f"{mean:.3f}")
    c2.metric("variance", f"{var:.3f}")
    c3.metric("std", f"{np.sqrt(var):.3f}")
    st.info(USES[dist], icon=":material/lightbulb:")


def _clt():
    left, right = st.columns([0.42, 0.58])
    with left:
        name = st.selectbox("base distribution (not Gaussian!)", list(DISTS), key="clt_base")
        n = st.slider("sample size n (averaged)", 1, 50, 5, key="p_n")
        seed = st.number_input("seed", 0, 9999, 0, key="p_seed")
        trials = 4000
    rng = np.random.default_rng(int(seed))
    samples = DISTS[name](rng, (trials, int(n)))
    means = samples.mean(axis=1)
    m, s = float(means.mean()), float(means.std())
    with right:
        fig, ax = plt.subplots(figsize=(4.7, 4.0))
        ax.hist(means, bins=40, density=True, color="#185FA5", alpha=0.75, label=f"mean of n={n}")
        if s > 1e-9:
            gx = np.linspace(means.min(), means.max(), 200)
            ax.plot(gx, np.exp(-(gx - m) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi)),
                    color="#C0507A", lw=2, label="Gaussian fit")
        ax.set_title("distribution of the sample mean")
        ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("n (averaged)", int(n))
    c2.metric("mean of means", f"{m:.3f}")
    c3.metric("std of means", f"{s:.3f}")
    st.info("At **n = 1** you see the base distribution's true (often skewed) shape. As **n "
            "grows**, the average becomes **Gaussian** (CLT) and its spread shrinks ~1/√n.",
            icon=":material/lightbulb:")


st.title("X3 · Probability — playground & lesson")
st.caption("Explore the key distributions (shape, mean, variance, where they're used), or "
           "watch the Central Limit Theorem turn any distribution's average into a Gaussian.")

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    mode = st.radio("playground", ["Distribution explorer", "Central Limit Theorem"],
                    horizontal=True, key="p_mode")
    st.divider()
    if mode == "Distribution explorer":
        _explorer()
    else:
        _clt()

with tab_theory:
    st.markdown(LESSON.theory)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mathprob")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
