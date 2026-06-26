"""Lesson content + a tiny self-check quiz engine for the GUI.

A Lesson bundles deep theory, a quiz, hands-on tasks, and references for one
concept. The Playground renders these alongside the interactive bench. Add more
lessons (perceptron, activations, ...) by appending to REGISTRY — the structure
is reusable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st


@dataclass
class Question:
    prompt: str
    options: list[str]
    answer: int          # index into options
    explanation: str


@dataclass
class Lesson:
    key: str
    title: str
    theory: str          # markdown (LaTeX with $...$ / $$...$$)
    quiz: list[Question]
    tasks: str           # markdown
    references: str       # markdown
    intro_blocks: list = field(default_factory=list)  # [{"md":..} | {"svg":..,"caption":..}]


def render_quiz(questions: list[Question], prefix: str) -> None:
    """Immediate-feedback quiz. Picking an option reveals correct/explanation."""
    top = st.container()
    if st.button("Reset answers", key=f"{prefix}_reset"):
        for i in range(len(questions)):
            st.session_state.pop(f"{prefix}_q{i}", None)
        st.rerun()

    correct = answered = 0
    for i, q in enumerate(questions):
        st.markdown(f"**Q{i + 1}. {q.prompt}**")
        choice = st.radio(
            "Your answer:", q.options, index=None,
            key=f"{prefix}_q{i}", label_visibility="collapsed",
        )
        if choice is not None:
            answered += 1
            if q.options.index(choice) == q.answer:
                correct += 1
                st.success(f"Correct — {q.explanation}")
            else:
                right = q.options[q.answer]
                st.error(f"Not quite. Answer: **{right}**. {q.explanation}")
        st.divider()

    with top:
        if answered:
            st.metric("Score", f"{correct}/{len(questions)}",
                      help=f"{answered} of {len(questions)} answered")
        else:
            st.caption("Pick an answer to get instant feedback. Nothing is graded — this is for you.")


def render_intro(blocks: list) -> None:
    """Render Part 0 intro blocks: markdown text interleaved with inline SVG figures."""
    for block in blocks:
        if "md" in block:
            st.markdown(block["md"])
        elif "svg" in block:
            st.markdown(
                f"<div style='text-align:center;margin:0.4rem 0'>{block['svg']}</div>",
                unsafe_allow_html=True,
            )
            if block.get("caption"):
                st.caption(block["caption"])


def render_lesson_content(lesson) -> None:
    """Render the 4 standard content tabs for a lesson (no interactive playground)."""
    t1, t2, t3, t4 = st.tabs(["📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"])
    with t1:
        st.markdown(lesson.theory)
    with t2:
        st.subheader("Self-check")
        st.caption("Instant feedback, no grading.")
        render_quiz(lesson.quiz, prefix=lesson.key)
    with t3:
        st.subheader("Tasks")
        st.markdown(lesson.tasks)
    with t4:
        st.subheader("Reading & references")
        st.markdown(lesson.references)


# ===========================================================================
# Lesson: the single artificial neuron
# ===========================================================================

_THEORY = r"""
## 1. What a neuron computes

A neuron maps an input vector $\mathbf{x}=(x_1,\dots,x_n)$ to a single scalar in
**two stages**:

$$ z = \mathbf{w}\cdot\mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b, \qquad a = \varphi(z) $$

- $z$ — the **pre-activation** ("logit", "net input"): an affine function of the inputs.
- $\varphi$ — the **activation** (the nonlinearity).
- $a$ — the neuron's **output**.

The two stages have very different jobs: the **linear** part ($z$) *mixes* the
inputs; the **nonlinear** part ($\varphi$) *decides / squashes*. Essentially all
of deep learning is alternating these two operations.

**Vectorized / batched.** For a layer of $u$ neurons, stack the weight vectors
into $W\in\mathbb{R}^{u\times n}$ and biases into $\mathbf b\in\mathbb{R}^{u}$:
$\mathbf z = W\mathbf x + \mathbf b$. For a batch of $m$ inputs
$X\in\mathbb{R}^{m\times n}$: $Z = XW^{\top}+\mathbf b$ — one matrix multiply
(why GPUs love neural nets).

## 2. The three ingredients

**Weights $\mathbf w$** — one per input. Sign = excite ($+$) or inhibit ($-$);
magnitude = how much that input matters. They are what the network *learns*.

**Bias $b$** — the offset/threshold. It lets the neuron output something nonzero
when all inputs are zero, and sets *how much evidence* is needed before the output
flips. *Bias trick:* append a constant input $x_0=1$ with weight $w_0=b$; then
$z=\tilde{\mathbf w}\cdot\tilde{\mathbf x}$ — bias is "just another weight".

**Activation $\varphi$** — squashes/thresholds $z$ into the output. Its shape
decides whether the neuron behaves like a hard switch, a soft probability, or a
rectifier (§5).

## 3. Four ways to *understand* one neuron

Same equation — four mental models, each illuminating something different.

**(a) A weighted vote / evidence accumulator.** Each input casts a weighted vote
$w_i x_i$; $b$ is a prior leaning; $z$ is the tally; $\varphi$ turns the tally
into a decision. Positive $w_i$ = "evidence *for*", negative = "evidence *against*".

**(b) A hyperplane (the geometry).** The set
$\{\mathbf x:\mathbf w\cdot\mathbf x+b=0\}$ is a hyperplane (a line in 2D):
- $\mathbf w$ is its **normal vector** — it points to the "+" side and sets the orientation.
- The **signed distance** of a point to it is $\dfrac{\mathbf w\cdot\mathbf x+b}{\lVert\mathbf w\rVert}$.
- It cuts space into two half-spaces ($z>0$, $z<0$).

So a neuron asks: *which side, and how far?* In the Playground, $\mathbf w$
rotates the boundary and $b$ slides it.

**(c) A template matcher / matched filter — the most physical view.** Write the
dot product in polar form:

$$ z=\mathbf w\cdot\mathbf x=\lVert\mathbf w\rVert\,\lVert\mathbf x\rVert\cos\theta $$

where $\theta$ is the angle between input and weight vector. For inputs of fixed
length, $z$ is **largest when $\mathbf x$ points the same way as $\mathbf w$**.
So the weight vector *is the pattern the neuron is tuned to* — its **preferred
stimulus**. A neuron is a correlation detector: it "lights up" for inputs that
look like its weights. (Visualize the first-layer weights of an image network and
you literally see edge/color templates.)

**(d) A soft logic gate.** With a sigmoid and large weights it approximates
AND/OR/NOT depending on $b$ — a *differentiable* logic gate (slide $b$ in the
Playground to turn AND into OR).

## 4. The physical picture

Map the math onto a (cartoon) physical neuron:

| symbol | physical reading |
|---|---|
| inputs $x_i$ | incoming signals — firing rates of upstream neurons / feature values |
| weights $w_i$ | **synaptic strengths / gains** ($+$ excitatory, $-$ inhibitory) |
| $\mathbf w\cdot\mathbf x$ | total synaptic drive — the "net input current" |
| bias $b$ | **excitability / threshold** — baseline readiness to fire |
| $z$ | "membrane potential" relative to threshold |
| $\varphi$ | the **f–I curve**: input current $\to$ output firing rate |
| $a$ | the neuron's firing rate / output signal |

The activation *is* the neuron's input–output response curve. A step = all-or-none
firing; a sigmoid = a saturating firing rate (there is a maximum rate); ReLU =
rate proportional to current above threshold, zero below. Weight *magnitude* is
gain; *learning* is adjusting synaptic strengths.

## 5. Activation functions — what they do and how they work

Three jobs, all essential:
1. **Inject nonlinearity** so depth means something (§8).
2. **Shape / bound the signal** — e.g. into a probability $(0,1)$ or a non-negative rate.
3. **Gate the gradient.** The derivative $\varphi'(z)$ is the neuron's **local
   gain** — how strongly a change in $z$ changes the output. Backprop multiplies
   these gains along every path, so $\varphi'$ decides whether the learning signal
   survives (§6).

**The catalogue** (switch between these in the Playground and watch the surface):

| $\varphi$ | formula | range | $\varphi'$ (gain) | character |
|---|---|---|---|---|
| step | $1$ if $z\ge0$ | $\{0,1\}$ | $0$ a.e. | hard switch; **no gradient → untrainable** by GD |
| sign | $\pm1$ | $\{-1,1\}$ | $0$ a.e. | bipolar step |
| linear | $z$ | $\mathbb{R}$ | $1$ | no squashing; regression output |
| sigmoid | $\dfrac{1}{1+e^{-z}}$ | $(0,1)$ | $\sigma(1-\sigma)\le0.25$ | smooth switch; **probability** |
| tanh | $\tanh z$ | $(-1,1)$ | $1-\tanh^2 z\le1$ | zero-centered sigmoid |
| ReLU | $\max(0,z)$ | $[0,\infty)$ | $1$ if $z>0$ else $0$ | rectifier; sparse; cheap |
| Leaky/PReLU | $\max(\alpha z, z)$ | $\mathbb{R}$ | $1$ or $\alpha$ | fixes "dead" ReLU |
| ELU | $z$ or $\alpha(e^{z}-1)$ | $(-\alpha,\infty)$ | smooth | mean-centered, robust |
| GELU / SiLU | $z\,\Phi(z)$ / $z\,\sigma(z)$ | $\gtrsim-0.3$ | smooth | **Transformer default** |
| softmax | $e^{z_i}/\sum_j e^{z_j}$ | $(0,1)$, $\sum=1$ | Jacobian | **competition** across a layer |

**Physical / statistical meaning of the important ones**

- **Sigmoid = log-odds → probability.** If $a=\sigma(z)$ is a probability $p$,
  then $z=\ln\frac{p}{1-p}$ — the pre-activation *is the log-odds*, and each $w_i$
  is the "weight of evidence" that input contributes (same quantity as logistic
  regression / naive Bayes). Shape: the logistic / Fermi–Dirac curve — a smooth
  threshold that **saturates** toward 0 and 1.
- **tanh** is a rescaled sigmoid, $\tanh z = 2\sigma(2z)-1$, but **zero-centered**
  — outputs average near 0, keeping the next layer's inputs balanced and gradients
  better-conditioned. Still saturates.
- **ReLU = a half-wave rectifier (an ideal diode).** It passes positive current
  and blocks negative — exactly a diode, and a fair model of a neuron's
  non-negative firing rate. Consequences: **sparse** activations (many exact
  zeros), constant gain $1$ on the active side (**gradients don't shrink**), and
  trivial compute. Failure mode: a unit stuck at $z<0$ for all data has gain $0$
  forever — a **dead ReLU**. Leaky/PReLU/ELU give the negative side a small slope
  to avoid that.
- **GELU / SiLU (Swish)** are smooth ReLUs that *softly* gate each input by how
  positive it is ($x \times$ probability-it-is-positive). Smoothness helps
  optimization; they are the default in modern Transformers.
- **softmax is not pointwise — it is a competition.** It normalizes a *vector* of
  scores into a probability distribution, so raising one output lowers the others
  (winner-take-most). Physically it is the **Boltzmann distribution**:
  $p_i=\dfrac{e^{z_i/T}}{\sum_j e^{z_j/T}}$, with $-z_i$ as energy and $T$ as
  **temperature**. $T\to0$ → argmax (one winner); large $T$ → uniform. (You meet
  this exact temperature again as the LLM sampling knob in Tier 5.)

## 6. Saturation, gain, and vanishing / exploding gradients

Backprop multiplies local gains $\varphi'$ along a path of depth $L$, so the
learning signal scales like a **product of gains**. With sigmoid
($\varphi'\le0.25$): through $L$ layers the gradient is at most $0.25^{L}$ — it
**vanishes** fast, and early layers barely learn. If gains are consistently $>1$,
gradients **explode**. ReLU's gain is exactly $1$ on the active side — which is
why ReLU-family activations + sensible initialization (He, e10) + normalization
made very deep nets trainable. **This is the single most important practical fact
about activations.**

## 7. How to choose an activation

**Hidden layers**
- **Default: ReLU** — cheap, sparse, non-saturating. Start here.
- **Transformers / modern deep nets: GELU or SiLU.**
- **Dead units hurting you: Leaky ReLU or ELU.**
- **Avoid sigmoid/tanh in deep hidden stacks** — they saturate and vanish
  gradients (tanh is still fine shallow, and as RNN state).

**Output layer — pick by what you are predicting**

| task | output activation | paired loss |
|---|---|---|
| regression (real value) | **linear** (none) | MSE |
| positive value (count, price) | softplus / exp | MSE / Poisson |
| bounded value $[-1,1]$ | tanh | MSE |
| binary classification | **sigmoid** | binary cross-entropy |
| multiclass, one label | **softmax** | cross-entropy |
| multi-label, many labels | **sigmoid per class** | BCE per class |

**Two rules of thumb**
1. **Match activation to loss.** sigmoid+BCE and softmax+cross-entropy give the
   clean gradient $\partial L/\partial z = a-y$ (no $\varphi'$ factor to stall
   learning). Never pair softmax with MSE.
2. **Hidden = keep gradients alive (ReLU-family); output = match the target's
   range.**

(RNN cells use sigmoid for *gates* — values in $(0,1)$ act as soft on/off valves —
and tanh for the *cell state*.)

## 8. Why the nonlinearity is non-negotiable

If $\varphi$ were linear, $\varphi(z)=\alpha z+\beta$, a stack of neurons would
compute an affine function of an affine function — still **affine**. So *any*
depth collapses to one linear layer: zero extra power. The nonlinearity is what
lets a network bend space and carve curved decision regions. (XOR, §9, is the
smallest proof that one linear cut isn't enough.)

## 9. Logic gates & linear separability — the neuron's "hello world"

Why does the Playground ship with **AND, OR, NAND, XOR** presets? Because the
2-input boolean functions are the **smallest, exact, fully-visualizable tests** of
what one neuron can and cannot do — only 4 points, no dataset, and you can *see*
the decision line.

A single neuron is **one straight cut**. It computes a gate exactly when that gate
is **linearly separable** — one line with all the 1-outputs on one side, all the
0-outputs on the other.

- **AND, OR, NAND, NOR are linearly separable** → a single neuron *can* represent
  each (set the weights by hand in e01; the perceptron *learns* them in e02). In the
  Playground, load **AND** and slide only the **bias** — it morphs into **OR** (same
  line, shifted).
- **XOR and XNOR are NOT linearly separable** → **no single neuron can ever
  represent them**, with any weights: their two "1" corners sit on opposite
  diagonals, and no straight line separates them. This is the famous **Minsky &
  Papert (1969)** result that helped trigger the first "AI winter".

**Why the gates matter (the point of the exercise):**
1. **They make "a neuron is a line" tangible** — and let you watch that line *fail*
   on XOR.
2. **XOR motivates depth.** The fix is a **hidden layer**: combine a few neurons
   (XOR = (A OR B) AND NOT(A AND B)) and the network bends space so XOR becomes
   separable. That one failure is *why* multilayer networks + backprop exist — the
   doorway to Tier 1 and beyond.
3. **Neurons can do logic at all.** McCulloch & Pitts (1943) first showed a neuron
   can act as a logic gate — the original bridge between brains and computation.
4. **NAND is functionally complete** — *any* boolean circuit can be built from NAND
   gates alone. So one neuron is a gate, and **layers of neurons can compute any
   logical function** (and far more): a neural net is, in principle, a universal
   computer.

Capacity footnote: the **VC dimension** of a linear classifier in $n$-D is $n+1$
(a 2D line shatters any 3 points in general position, but not 4) — and XOR's 4
points are exactly the case one line can't handle. Of the 16 two-input gates, **14
are linearly separable; only XOR and XNOR are not** (Cover, 1965).

## 10. It's the atom of classical ML too

| neuron | + loss / rule | = classical model |
|---|---|---|
| linear | mean squared error | **linear regression** |
| sigmoid | binary cross-entropy | **logistic regression** |
| sign | perceptron update | **perceptron** |
| linear | max-margin (hinge) | **linear SVM** |

So "single neuron" isn't a toy — it's logistic regression with a learnable threshold.

## 11. Cost, learning, and numerics

- **Cost:** one neuron is $O(n)$ multiply-adds; a layer on a batch is one matmul,
  $O(m\,n\,u)$; params $=n\cdot u+u$.
- **Learning the weights** (you set them by hand in e01): the **perceptron rule**
  (e02, converges iff linearly separable) or **gradient descent + backprop**
  (Tier 1, general).
- **Numerics:** naive $1/(1+e^{-z})$ overflows for very negative $z$ — the lab's
  `core/activations.py` splits by sign for stability. Weight **scale** matters:
  too big saturates $\varphi$, too small starves the signal; initialization (e10)
  controls this.

## 12. Biological inspiration (and the honest caveat)

Dendrites ≈ inputs, synaptic strengths ≈ weights, the soma sums them, the axon
"fires" past threshold ≈ activation (rate coding). But real neurons **spike** in
time, are stochastic, and have rich dendritic dynamics. The artificial neuron is a
deliberate, useful *caricature* — good enough to build intelligence from, not a
faithful biological model.

## 13. Worked examples — one neuron in the wild

The same $a=\varphi(\mathbf w\cdot\mathbf x+b)$ already does useful work on its own
(it *is* logistic / linear regression). A few single-neuron models:

- **Loan approval** — $\mathbf x=[\text{income},\,\text{debt},\,\text{credit score}]$,
  sigmoid output = probability of repayment. Positive weights on income/credit,
  negative on debt; the bias sets how cautious the lender is.
- **Spam filter** — features like "contains the word *free*", "number of links",
  "ALL-CAPS ratio"; sigmoid output = $P(\text{spam})$. Each weight is a feature's
  "spamminess" — literally a log-odds (§5).
- **House price** — a *linear* neuron over $[\text{area},\,\text{bedrooms},\,
  \text{age}]$ → a price. (That's linear regression — see the ML module.)
- **Medical risk** — $[\text{age},\,\text{BMI},\,\text{glucose}]\to P(\text{diabetes})$.

**A tiny numeric trace (spam).** Weights $\mathbf w=[2.0,\,1.5,\,-1.0]$ for
[has "free", #links, from-known-contact], bias $b=-1$, input $\mathbf x=[1,\,3,\,0]$:
$$z = 2(1)+1.5(3)-1(0)-1 = 5.5,\qquad a=\sigma(5.5)\approx 0.996.$$
99.6% sure it's spam — the neuron *accumulated evidence*: "free" and the three
links pushed $z$ up, with no trusted-sender signal to pull it down.

## 14. From one neuron to today's AI

One neuron draws one boundary. The magic is **stacking**: a network of these units
with nonlinear activations is a **universal approximator** — it can represent
essentially any function. Add structure for the data type and you get every modern
system:

| add this structure | architecture | powers (real apps) |
|---|---|---|
| many layers (MLP) | deep network | tabular prediction, fraud, ranking |
| weight sharing over space | **CNN** | vision: face unlock, medical imaging, self-driving perception |
| recurrence / memory | RNN, LSTM | early speech & translation |
| **attention** | **Transformer** | **LLMs (Claude, GPT), translation, code (Copilot)** |
| learned vectors | embeddings | recommendation (Netflix, Spotify), semantic search, RAG |
| iterative denoising | diffusion | image generation (DALL·E, Midjourney, Stable Diffusion) |

**An LLM in one breath:** it is **billions of these exact neurons** in Transformer
blocks. Each block is mostly two things you already know — an **attention** step
(weighted sums deciding what to attend to) and an **MLP** (stacked neurons with a
**GELU** activation, §5). The final layer emits one score per vocabulary word; a
**softmax** (§5, with its **temperature** knob) turns those into next-token
probabilities; the model samples one, appends it, and repeats. Training is
**gradient descent + backprop** (§6) over trillions of words.

So nothing here is "toy". The neuron in the Playground, the activation you pick, the
softmax temperature, the match-loss-to-output rule — those are *literally* the parts
of the systems you use daily. This lab's arc (single neuron → MLP → attention → a
small GPT) is the real path from this panel to ChatGPT, just scaled up.
"""

_TASKS = r"""
### Warm-up — in the Playground tab
1. Load **AND (step)**, then drag **only the bias** from $-1.5$ toward $-0.5$.
   Watch it become **OR**. In one sentence: why does only $b$ change the gate?
2. Find $\mathbf w, b$ by hand for **NAND** and **NOR**; verify in the truth table.
3. Make a **sigmoid** neuron behave almost like a step: how large must
   $\lVert\mathbf w\rVert$ be so the four corner outputs are within $0.01$ of $0/1$?

### Pencil & paper
4. Write the boundary line for $\mathbf w=[1,2],\,b=-3$. Where does it cross each axis?
5. Prove XOR is not linearly separable (2 lines of argument).
6. Name the **2 of 16** two-input boolean functions a single neuron can't represent.
7. Derive the bias trick: show $\mathbf w\cdot\mathbf x+b=\tilde{\mathbf w}\cdot\tilde{\mathbf x}$.
8. Compute the signed distance from $\mathbf x=[2,0]$ to the boundary of
   $\mathbf w=[1,1],\,b=-1$.

### Activations
9. For each setting name the activation you'd pick and **why**: (a) hidden layer of
   a 30-layer net, (b) binary classifier output, (c) 10-class output, (d) bounded
   output in $[-1,1]$.
10. Show algebraically that $\tanh z = 2\sigma(2z)-1$ (so tanh is a rescaled sigmoid).
11. Confirm $\sigma'(z)$ peaks at $z=0$ with value $0.25$; then explain in one
    sentence why a 20-layer all-sigmoid net barely trains.

### Code — extend `core/`, keep tests green
12. Add `signed_distance(self, x)` to `core/neuron.Neuron` returning
    $(\mathbf w\cdot\mathbf x+b)/\lVert\mathbf w\rVert$; add a test in `tests/`.
13. Write a `boolean_fit(truth_table)` helper that brute-forces integer weights in
    $[-2,2]$ to realize a 2-input gate; confirm which gates are realizable.
14. **Bridge to e02:** implement the perceptron learning rule and train AND/OR
    from a random init.

### Stretch
15. Estimate capacity empirically: for $n=2$, randomly label $k$ points and measure
    the fraction a single neuron can separate as $k$ grows $2\to6$. Relate to
    VC dimension $=n+1=3$.
"""

_REFERENCES = r"""
### Foundational papers
- **McCulloch & Pitts (1943)** — *A Logical Calculus of the Ideas Immanent in
  Nervous Activity.* The first artificial neuron.
- **Rosenblatt (1958)** — *The Perceptron.* The first learning rule.
- **Minsky & Papert (1969)** — *Perceptrons.* The XOR / linear-separability limit.
- **Cover (1965)** — *Geometrical and Statistical Properties of Systems of Linear
  Inequalities* — capacity of linear separation.

### Books (with the right chapter)
- Nielsen — *Neural Networks and Deep Learning*, ch. 1 — [free online](http://neuralnetworksanddeeplearning.com/chap1.html).
- Goodfellow, Bengio & Courville — *Deep Learning*, ch. 6 — [deeplearningbook.org](https://www.deeplearningbook.org/).
- Bishop — *Pattern Recognition and ML*, ch. 4 (linear models for classification).
- Hastie, Tibshirani & Friedman — *Elements of Statistical Learning*, ch. 4.
- Hertz, Krogh & Palmer — *Introduction to the Theory of Neural Computation* (1991)
  — the classic graduate theory text (Hopfield nets, statistical mechanics).

### Courses & video
- **3Blue1Brown** — [neural networks series](https://www.3blue1brown.com/topics/neural-networks) (intuition).
- **Karpathy** — [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) + `micrograd`.
- **d2l.ai** — [Dive into Deep Learning](https://d2l.ai) (math + runnable code).

### In this lab
- Infobase: `infobase/00_foundations/single-neuron.md`, `activation-functions.md`,
  `perceptron.md`.
- Experiments: **e01** (this), **e02** (perceptron), **e03** (XOR fails).
- See `papers/README.md` for the full annotated reading list.
"""

_QUIZ = [
    Question(
        "In $z=\\mathbf{w}\\cdot\\mathbf{x}+b$, what does the bias $b$ do?",
        ["Scales the inputs", "Shifts the decision boundary (the threshold)",
         "Is the activation function", "Normalizes the output"],
        1,
        "b slides the hyperplane w·x+b=0; it's the threshold, equivalently a weight on a constant +1 input.",
    ),
    Question(
        "Why must a neuron include a nonlinear activation $\\varphi$?",
        ["To train faster", "To bound the weights",
         "Otherwise stacked neurons collapse to a single linear map",
         "Purely for biological realism"],
        2,
        "A composition of affine functions is affine, so without nonlinearity any depth has the power of one linear layer.",
    ),
    Question(
        "Geometrically, $\\mathbf{w}\\cdot\\mathbf{x}+b=0$ describes…",
        ["A single point", "A hyperplane separating two half-spaces",
         "A circle", "The activation curve"],
        1,
        "It's a hyperplane (line in 2D); w is its normal vector and it splits space into z>0 and z<0.",
    ),
    Question(
        "Given $\\mathbf{w}=[2,-1],\\,b=0.5,\\,\\mathbf{x}=[1,1]$, what is $z$?",
        ["0.5", "1.5", "2.5", "-0.5"],
        1,
        "z = 2·1 + (-1)·1 + 0.5 = 1.5.",
    ),
    Question(
        "With a step activation, which gate can a single neuron NOT compute?",
        ["AND", "OR", "XOR", "NAND"],
        2,
        "XOR is not linearly separable — no single line separates its classes (Minsky & Papert, 1969).",
    ),
    Question(
        "The maximum value of the sigmoid derivative $\\sigma'(z)$ is…",
        ["1", "0.5", "0.25 (at z=0)", "unbounded"],
        2,
        "At z=0, σ=0.5 so σ'=0.5·0.5=0.25. This small max is why deep sigmoid stacks vanish gradients.",
    ),
    Question(
        "A single sigmoid neuron trained with binary cross-entropy is equivalent to…",
        ["Linear regression", "Logistic regression", "k-nearest neighbors", "A decision tree"],
        1,
        "Sigmoid output + BCE loss is exactly logistic regression.",
    ),
    Question(
        "Increasing $\\lVert\\mathbf{w}\\rVert$ (same direction) makes a sigmoid neuron's transition…",
        ["Flatter", "Sharper, closer to a step", "Unchanged", "Inverted"],
        1,
        "Larger weights scale z, pushing outputs toward 0/1 — the boundary gets sharper. Try it in the Playground.",
    ),
    Question(
        "How is the bias absorbed into the weight vector (the 'bias trick')?",
        ["Add a constant input x₀=1 with weight w₀=b", "Multiply all weights by b",
         "Remove the activation", "Set b=0"],
        0,
        "With x̃=(1,x₁,…) and w̃=(b,w₁,…), z = w̃·x̃ — bias becomes just another weight.",
    ),
    Question(
        "The VC dimension (capacity) of a linear classifier in $n$-D input space is…",
        ["n", "n+1", "2ⁿ", "infinite"],
        1,
        "n+1: in 2D (n=2) a line can shatter 3 points but not 4.",
    ),
    Question(
        "Who introduced the perceptron learning rule?",
        ["McCulloch & Pitts", "Rosenblatt", "Minsky & Papert", "Rumelhart et al."],
        1,
        "Frank Rosenblatt, 1958. McCulloch & Pitts (1943) modeled the neuron; Rosenblatt added learning.",
    ),
    Question(
        "Writing $z=\\lVert\\mathbf{w}\\rVert\\,\\lVert\\mathbf{x}\\rVert\\cos\\theta$, an input of fixed "
        "length drives the neuron most strongly when it…",
        ["is orthogonal to w", "aligns in direction with the weight vector w",
         "points opposite to w", "has all-positive entries"],
        1,
        "z is maximal at cosθ=1, i.e. x parallel to w — the weight vector is the neuron's preferred stimulus / template.",
    ),
    Question(
        "Why is ReLU usually preferred over sigmoid in deep hidden layers?",
        ["It outputs probabilities", "It is zero-centered",
         "Its gain is 1 on the active side, so gradients don't vanish with depth",
         "It saturates on both sides"],
        2,
        "ReLU's derivative is 1 for z>0, so chained gains stay O(1); sigmoid's ≤0.25 gain shrinks as 0.25^L.",
    ),
    Question(
        "A 'dead' ReLU neuron is one that…",
        ["fires for every input", "has z<0 for all data, so its output and gradient are stuck at 0",
         "has exploding weights", "uses a negative bias only"],
        1,
        "If z<0 on all inputs, output=0 and gradient=0 forever — it can't recover. Leaky ReLU/ELU add a negative slope to prevent it.",
    ),
    Question(
        "In softmax $p_i=e^{z_i/T}/\\sum_j e^{z_j/T}$, increasing the temperature $T$…",
        ["makes the distribution sharper (closer to argmax)",
         "makes the distribution more uniform", "has no effect", "always picks the smallest score"],
        1,
        "Higher T flattens the distribution; T→0 approaches argmax. The same knob as LLM sampling temperature (Tier 5).",
    ),
    Question(
        "For single-label classification over 10 mutually exclusive classes, the output layer should use…",
        ["sigmoid per class + MSE", "softmax + cross-entropy", "tanh + MSE", "ReLU + BCE"],
        1,
        "Mutually-exclusive classes → softmax (a distribution summing to 1) paired with cross-entropy.",
    ),
    Question(
        "For a sigmoid neuron with output probability $p$, the pre-activation $z$ equals…",
        ["p itself", "the log-odds ln(p/(1−p))", "1−p", "p²"],
        1,
        "Inverting p=σ(z) gives z=ln(p/(1−p)) — the logit / log-odds; each weight is a 'weight of evidence'.",
    ),
    Question(
        "Stacking neurons with nonlinear activations can approximate…",
        ["only linear functions", "essentially any continuous function (universal approximation)",
         "only boolean functions", "nothing more than one neuron"],
        1,
        "The universal approximation theorem: a network with a hidden layer can approximate any continuous function given enough units.",
    ),
    Question(
        "An LLM turns its final-layer scores into next-token probabilities using…",
        ["ReLU", "softmax (with a temperature)", "a step function", "the raw bias term"],
        1,
        "Softmax over the vocabulary gives a probability per token; temperature controls how peaked the choice is.",
    ),
    Question(
        "Why does the lab use logic gates (AND, OR, XOR) to study a single neuron?",
        ["they are hard to compute",
         "they are the smallest exact, visualizable test of what one linear neuron can/can't represent",
         "they require large datasets", "they only run on GPUs"],
        1,
        "Four points, no dataset, a visible decision line — the perfect minimal probe of linear separability.",
    ),
    Question(
        "NAND gates are special in computing because…",
        ["they are the fastest gate",
         "they are functionally complete — any boolean circuit can be built from NANDs",
         "they need no inputs", "only brains can compute them"],
        1,
        "NAND is universal, so layered neurons (each able to act as a NAND) can in principle compute any logic.",
    ),
]

# --- Part 0: biological neuron — text ---------------------------------------

_BIO_INTRO = r"""
## 0. The biological neuron — where the idea comes from

Before any math: the artificial neuron is a deliberate *cartoon* of a real brain
cell. It throws away almost all the biology but keeps one thing — the
**computational role**. Seeing the real cell first makes every later symbol
($\mathbf w$, $b$, $\varphi$) feel inevitable.
"""

_BIO_HOW = r"""
### How a real neuron works

- **Dendrites** — branched fibers that *receive* signals from up to ~10,000 other
  neurons. (These are the inputs.)
- **Soma (cell body)** — *integrates* the incoming signals; each nudges the
  membrane voltage up (**excitatory**) or down (**inhibitory**).
- **Threshold & the spike** — if the summed voltage at the axon hillock crosses a
  threshold (≈ −55 mV), the neuron *fires*: a sharp **all-or-none** pulse, the
  **action potential**. Below threshold → nothing; above → a full spike. That is
  the original "step function".
- **Axon** — the output cable. **Myelin** insulation lets the spike race along
  (up to ~120 m/s), jumping between gaps (nodes of Ranvier).
- **Synapse** — the junction where one neuron's axon terminal meets the next
  neuron's dendrite. The spike triggers release of **neurotransmitter** molecules
  across a tiny gap; they bind receptors and create a voltage change in the next
  cell.
- **Rate coding** — neurons mostly signal by *how often* they spike (firing rate),
  not by a single spike. That firing-rate-vs-input curve is the biological
  ancestor of the **activation function**.
"""

_BIO_KNOWLEDGE = r"""
### How connections store knowledge

The brain's knowledge is **not** in the neurons — it is in the **pattern and
strength of the connections** (synapses) between them. Three big ideas:

- **Synaptic strength = a weight.** A strong synapse passes a large signal, a weak
  one almost none — exactly the artificial weight $w_i$. The human brain has
  ~$10^{11}$ neurons and ~$10^{14}$–$10^{15}$ synapses: those synapses are the
  "parameters".
- **Hebbian learning — "neurons that fire together, wire together."** When neuron
  A reliably helps fire neuron B, the A→B synapse *strengthens* (long-term
  potentiation); uncorrelated activity *weakens* it. Learning = changing synaptic
  strengths — the biological ancestor of gradient descent updating weights.
- **Distributed representation.** A concept ("grandmother", "the letter A") is not
  one neuron — it is a *pattern of activity* across many neurons, stored as a
  *pattern of weights*. Each neuron takes part in many concepts. That is why the
  brain degrades gracefully when cells die, and why "knowledge" is smeared across
  the connections rather than filed in one place.
"""

_BIO_REASONING = r"""
### Where "reasoning" comes from

There is no central CPU. "Computation" is just **activity propagating through
weighted connections**:

- Sensory neurons fire → signals flow through layers of synapses → each downstream
  neuron fires if its *weighted* input crosses threshold → … → output neurons
  drive a response. **Inference = this forward propagation.**
- Recognizing a face, completing a pattern, recalling a memory from a partial cue
  are all the network *settling* into an activation pattern dictated by its
  weights. Recurrent loops let activity persist (working memory) and settle into
  stable **attractor** states — the idea behind Hopfield networks and associative
  memory.
- So: **memory = the weights; thinking = the dynamics of activation over those
  weights.** Change the weights (learn) and you change both what the system knows
  *and* how it reasons.
"""

_BIO_COMPARE = r"""
### Artificial vs. biological — the honest comparison

| biological | artificial | kept? |
|---|---|---|
| dendrite signals (firing rates) | inputs $x_i$ | kept (as real numbers) |
| synaptic strength | weight $w_i$ | kept |
| excitatory / inhibitory | $w_i>0$ / $w_i<0$ | kept |
| soma summation | $\sum w_i x_i$ | kept |
| firing threshold | bias $b$ | kept |
| firing-rate response (f–I curve) | activation $\varphi$ | kept (smoothed) |
| spike train in time | one number $a$ | **dropped** (no time) |
| Hebbian plasticity | gradient descent | replaced |
| ~$10^{15}$ synapses, ~20 watts | billions of params, kilowatts | very different |

**Kept:** integrate weighted inputs → compare to a threshold → emit a graded
output; learn by adjusting connection strengths. **Dropped:** spikes and timing,
ion-channel biophysics, dendritic computation, neuromodulators, and the brain's
astonishing energy efficiency.

A real neuron *inspired* the model, but the artificial neuron abstracts its
**computational job**, not the cell. The power of deep learning comes from
stacking millions of these caricatures and training them — not from biological
realism. (For the dynamics-of-activation view in depth, your 1994 course text —
Hertz, Krogh & Palmer — is still excellent.)
"""

# --- Part 0: biological neuron — figures (self-contained SVG) ----------------

_SVG_NEURON = """
<svg viewBox="0 0 760 300" style="width:100%;height:auto;max-width:780px" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Anatomy of a biological neuron">
  <rect x="1" y="1" width="758" height="298" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/>
  <g stroke="#1D6FB8" stroke-width="2.5" fill="none" stroke-linecap="round">
    <path d="M180 150 L92 92 M126 116 L72 74 M126 116 L70 124"/>
    <path d="M178 150 L80 150 M112 150 L60 134 M112 150 L60 166"/>
    <path d="M180 150 L96 210 M128 188 L74 202 M128 188 L84 236"/>
  </g>
  <ellipse cx="215" cy="150" rx="48" ry="40" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="2.5"/>
  <circle cx="215" cy="150" r="17" fill="#C98A3A" stroke="#8A5E22" stroke-width="2"/>
  <line x1="262" y1="150" x2="560" y2="150" stroke="#5B8FC2" stroke-width="3"/>
  <g fill="#CFE3F5" stroke="#5B8FC2" stroke-width="2">
    <rect x="300" y="138" width="64" height="24" rx="12"/>
    <rect x="380" y="138" width="64" height="24" rx="12"/>
    <rect x="460" y="138" width="64" height="24" rx="12"/>
  </g>
  <g stroke="#1D9E75" stroke-width="2.5" fill="none" stroke-linecap="round">
    <path d="M560 150 L640 110 M560 150 L652 150 M560 150 L640 196"/>
  </g>
  <circle cx="645" cy="106" r="7" fill="#1D9E75"/>
  <circle cx="658" cy="150" r="7" fill="#1D9E75"/>
  <circle cx="645" cy="200" r="7" fill="#1D9E75"/>
  <g stroke="#6B6A66" stroke-width="1.5" fill="#6B6A66">
    <line x1="300" y1="272" x2="520" y2="272"/>
    <path d="M520 272 l-10 -5 l0 10 z"/>
  </g>
  <g fill="#33312E" font-family="sans-serif" font-size="13">
    <text x="58" y="52">dendrites (inputs)</text>
    <text x="180" y="236" text-anchor="middle">soma (cell body)</text>
    <text x="215" y="118" text-anchor="middle" font-size="11" fill="#6B6A66">nucleus</text>
    <text x="332" y="128" text-anchor="middle">myelin sheath</text>
    <text x="372" y="180" text-anchor="middle" font-size="11" fill="#6B6A66">node</text>
    <text x="430" y="200" text-anchor="middle">axon</text>
    <text x="648" y="90" text-anchor="middle">axon terminals</text>
    <text x="410" y="290" text-anchor="middle" font-size="11" fill="#6B6A66">signal flow</text>
  </g>
</svg>
"""

_SVG_SYNAPSE = """
<svg viewBox="0 0 760 320" style="width:100%;height:auto;max-width:780px" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A synapse transmitting a signal">
  <rect x="1" y="1" width="758" height="318" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/>
  <ellipse cx="205" cy="175" rx="155" ry="95" fill="#CFE3F5" stroke="#5B8FC2" stroke-width="2"/>
  <g fill="#7FB0E0" stroke="#5B8FC2" stroke-width="1.5">
    <circle cx="250" cy="140" r="12"/>
    <circle cx="272" cy="182" r="12"/>
    <circle cx="246" cy="216" r="12"/>
    <circle cx="305" cy="162" r="10"/>
  </g>
  <g fill="#C0507A">
    <circle cx="362" cy="150" r="5"/><circle cx="378" cy="172" r="5"/>
    <circle cx="366" cy="196" r="5"/><circle cx="388" cy="140" r="5"/>
    <circle cx="392" cy="186" r="5"/>
  </g>
  <rect x="418" y="70" width="55" height="210" rx="18" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="2"/>
  <g fill="#9A6A2A">
    <rect x="408" y="128" width="14" height="16" rx="3"/>
    <rect x="408" y="165" width="14" height="16" rx="3"/>
    <rect x="408" y="202" width="14" height="16" rx="3"/>
  </g>
  <line x1="473" y1="175" x2="560" y2="175" stroke="#9A6A2A" stroke-width="3"/>
  <g>
    <text x="602" y="78" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#33312E">action potential</text>
    <line x1="540" y1="150" x2="668" y2="150" stroke="#B7B6B0" stroke-width="1"/>
    <polyline points="540,150 588,150 595,150 600,95 606,150 668,150" fill="none" stroke="#1D6FB8" stroke-width="2"/>
  </g>
  <g font-family="sans-serif" font-size="13" fill="#33312E">
    <text x="205" y="302" text-anchor="middle">presynaptic terminal</text>
    <text x="272" y="108" text-anchor="middle" font-size="11" fill="#6B6A66">vesicles</text>
    <text x="378" y="246" text-anchor="middle" font-size="11" fill="#C0507A">neurotransmitters</text>
    <text x="378" y="58" text-anchor="middle" font-size="11" fill="#6B6A66">cleft</text>
    <text x="505" y="302" text-anchor="middle">postsynaptic neuron</text>
    <text x="400" y="112" text-anchor="end" font-size="11" fill="#6B6A66">receptors</text>
  </g>
</svg>
"""

_SVG_MAP = """
<svg viewBox="0 0 760 270" style="width:100%;height:auto;max-width:780px" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Mapping between biological and artificial neuron parts">
  <rect x="1" y="1" width="758" height="268" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/>
  <text x="26" y="82" transform="rotate(-90 26 82)" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#9A6A2A">biological</text>
  <text x="26" y="192" transform="rotate(-90 26 192)" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#1D6FB8">artificial</text>
  <g fill="#EFD3AE" stroke="#9A6A2A" stroke-width="2">
    <rect x="100" y="58" width="100" height="44" rx="8"/>
    <rect x="235" y="58" width="100" height="44" rx="8"/>
    <rect x="370" y="58" width="100" height="44" rx="8"/>
    <rect x="505" y="58" width="100" height="44" rx="8"/>
    <rect x="640" y="58" width="100" height="44" rx="8"/>
  </g>
  <g fill="#CFE3F5" stroke="#5B8FC2" stroke-width="2">
    <rect x="100" y="168" width="100" height="44" rx="8"/>
    <rect x="235" y="168" width="100" height="44" rx="8"/>
    <rect x="370" y="168" width="100" height="44" rx="8"/>
    <rect x="505" y="168" width="100" height="44" rx="8"/>
    <rect x="640" y="168" width="100" height="44" rx="8"/>
  </g>
  <g font-family="sans-serif" font-size="12" fill="#5A3E14" text-anchor="middle">
    <text x="150" y="85">dendrites</text>
    <text x="285" y="85">synapses</text>
    <text x="420" y="85">soma sum</text>
    <text x="555" y="85">threshold</text>
    <text x="690" y="85">spike</text>
  </g>
  <g font-family="sans-serif" font-size="12" fill="#0C447C" text-anchor="middle">
    <text x="150" y="195">inputs xᵢ</text>
    <text x="285" y="195">weights wᵢ</text>
    <text x="420" y="195">Σ wᵢxᵢ + b</text>
    <text x="555" y="195">activation φ</text>
    <text x="690" y="195">output a</text>
  </g>
  <g stroke="#C8C7C0" stroke-width="1.5" stroke-dasharray="3 3">
    <line x1="150" y1="102" x2="150" y2="168"/>
    <line x1="285" y1="102" x2="285" y2="168"/>
    <line x1="420" y1="102" x2="420" y2="168"/>
    <line x1="555" y1="102" x2="555" y2="168"/>
    <line x1="690" y1="102" x2="690" y2="168"/>
  </g>
  <g fill="#9C9B95" font-family="sans-serif" font-size="16" text-anchor="middle">
    <text x="150" y="140">≈</text><text x="285" y="140">≈</text><text x="420" y="140">≈</text>
    <text x="555" y="140">≈</text><text x="690" y="140">≈</text>
  </g>
  <g stroke="#B0AFA8" stroke-width="1.5" fill="#B0AFA8">
    <line x1="200" y1="80" x2="231" y2="80"/><path d="M235 80 l-8 -4 l0 8 z"/>
    <line x1="335" y1="80" x2="366" y2="80"/><path d="M370 80 l-8 -4 l0 8 z"/>
    <line x1="470" y1="80" x2="501" y2="80"/><path d="M505 80 l-8 -4 l0 8 z"/>
    <line x1="605" y1="80" x2="636" y2="80"/><path d="M640 80 l-8 -4 l0 8 z"/>
    <line x1="200" y1="190" x2="231" y2="190"/><path d="M235 190 l-8 -4 l0 8 z"/>
    <line x1="335" y1="190" x2="366" y2="190"/><path d="M370 190 l-8 -4 l0 8 z"/>
    <line x1="470" y1="190" x2="501" y2="190"/><path d="M505 190 l-8 -4 l0 8 z"/>
    <line x1="605" y1="190" x2="636" y2="190"/><path d="M640 190 l-8 -4 l0 8 z"/>
  </g>
</svg>
"""

_BIO_BLOCKS = [
    {"md": _BIO_INTRO},
    {"svg": _SVG_NEURON,
     "caption": "A biological neuron: dendrites collect inputs, the soma integrates them, and the axon carries an all-or-none spike to other neurons."},
    {"md": _BIO_HOW},
    {"svg": _SVG_SYNAPSE,
     "caption": "The synapse. An arriving spike releases neurotransmitters across the cleft; receptors convert them into a voltage change in the next neuron. Synaptic strength = the size of that change = the 'weight'."},
    {"md": _BIO_KNOWLEDGE},
    {"md": _BIO_REASONING},
    {"svg": _SVG_MAP,
     "caption": "The correspondence. The artificial neuron keeps the computation — weighted sum, threshold, graded output, learning by changing connection strengths — and drops the biophysics and timing."},
    {"md": _BIO_COMPARE},
]


SINGLE_NEURON = Lesson(
    key="single_neuron",
    title="The single artificial neuron",
    theory=_THEORY,
    quiz=_QUIZ,
    tasks=_TASKS,
    references=_REFERENCES,
    intro_blocks=_BIO_BLOCKS,
)

REGISTRY = {SINGLE_NEURON.key: SINGLE_NEURON}
