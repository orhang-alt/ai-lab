"""ML module lessons M3–M7. Reuses the Lesson/Question/quiz engine from lessons.py."""

from __future__ import annotations

from lessons import Lesson, Question

# ===========================================================================
# M3 — Trees & ensembles
# ===========================================================================

_M3_THEORY = r"""
## 1. The idea — a flowchart of yes/no questions

A **decision tree** predicts by asking a sequence of simple threshold questions —
*"is glucose > 120? if yes, is BMI > 30? ..."* — until it reaches a **leaf** that gives
the answer. Each internal node is an **axis-aligned split** ($x_j \le t$), so the tree
chops feature space into **rectangular boxes**, each labeled by the majority class
(classification) or the mean (regression) of the training points inside it.

The tree (left) *is* a recipe for slicing the feature space into boxes (right) — each
leaf is one box:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 560 270" style="width:100%;max-width:560px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Left: a decision tree of threshold questions with three leaves A, B, C. Right: the same rules chop the 2D feature space into three axis-aligned rectangles labeled A, B, C."><rect x="1" y="1" width="558" height="268" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g stroke="#9C9B95" stroke-width="1.5"><line x1="155" y1="64" x2="80" y2="108"/><line x1="157" y1="64" x2="234" y2="108"/><line x1="238" y1="142" x2="212" y2="178"/><line x1="240" y1="142" x2="276" y2="178"/></g><g font-family="sans-serif" font-size="10" fill="#6B6A66"><text x="100" y="92">yes</text><text x="202" y="92">no</text><text x="206" y="166">yes</text><text x="268" y="166">no</text></g><rect x="110" y="30" width="94" height="34" rx="6" fill="#FFFFFF" stroke="#5B8FC2" stroke-width="1.6"/><text x="157" y="51" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#0C447C">x₁ &lt; 5 ?</text><rect x="44" y="108" width="68" height="34" rx="6" fill="#D7EFE5" stroke="#1D9E75" stroke-width="1.6"/><text x="78" y="129" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#0E5E45">A</text><rect x="190" y="108" width="98" height="34" rx="6" fill="#FFFFFF" stroke="#5B8FC2" stroke-width="1.6"/><text x="239" y="129" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#0C447C">x₂ &lt; 6 ?</text><rect x="184" y="178" width="56" height="32" rx="6" fill="#F3E2C7" stroke="#9A6A2A" stroke-width="1.6"/><text x="212" y="198" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#5A3E14">B</text><rect x="248" y="178" width="56" height="32" rx="6" fill="#E2E0F5" stroke="#7F77DD" stroke-width="1.6"/><text x="276" y="198" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#3F3A8C">C</text><rect x="330" y="40" width="100" height="170" fill="#D7EFE5"/><rect x="430" y="130" width="100" height="80" fill="#F3E2C7"/><rect x="430" y="40" width="100" height="90" fill="#E2E0F5"/><rect x="330" y="40" width="200" height="170" fill="none" stroke="#C9C8C1"/><line x1="430" y1="40" x2="430" y2="210" stroke="#33312E" stroke-width="1.6"/><line x1="430" y1="130" x2="530" y2="130" stroke="#33312E" stroke-width="1.6"/><g font-family="sans-serif" font-size="14" text-anchor="middle"><text x="380" y="130" fill="#0E5E45">A</text><text x="480" y="176" fill="#5A3E14">B</text><text x="480" y="92" fill="#3F3A8C">C</text></g><text x="430" y="230" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#9C9B95">x₁ →</text><text x="322" y="128" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#9C9B95" transform="rotate(-90 322 128)">x₂ →</text><g font-family="sans-serif" font-size="11" fill="#6B6A66" text-anchor="middle"><text x="155" y="250">the rules (a tree)</text><text x="430" y="250">the regions (feature space)</text></g></svg></div>

Two things make trees special among the models so far:
- the boundary is **non-linear** (a staircase of rectangles), so a single tree can
  solve problems a line can't (e.g. XOR — try the Playground);
- it is **interpretable** — you can literally read the decision rules.

## 2. How a tree is built (CART)

Finding the globally optimal tree is NP-hard, so CART builds one **greedily**,
top-down:
1. At the current node, **scan every feature and every candidate threshold**.
2. Keep the split that most **reduces impurity** (below).
3. Send the data into the two children and **recurse** on each.
4. Stop when a rule fires (depth limit, too few samples, or the node is pure).

Greedy ≠ optimal, but it's fast and works remarkably well in practice.

## 3. Impurity & information gain

A node is *pure* when all its points share one class. We score a split by how much it
**reduces impurity**:
$$ \text{gain} = I(\text{parent}) - \tfrac{n_L}{n}I(\text{left}) - \tfrac{n_R}{n}I(\text{right}). $$

- **Gini impurity** $I = 1 - \sum_c p_c^2$ (the CART default; cheap).
- **Entropy** $I = -\sum_c p_c\log_2 p_c$ (gain = *information gain*).

Both are 0 for a pure node and maximal for a 50/50 mix.

**Worked split.** A node with class counts $[8,2]$ has $p=(0.8,0.2)$, so
$\text{Gini} = 1-(0.8^2+0.2^2) = 0.32$. Split it into $[6,0]$ (pure, Gini 0) and
$[2,2]$ (Gini $1-2\cdot0.5^2 = 0.5$). Weighted child impurity
$= \tfrac{6}{10}\cdot0 + \tfrac{4}{10}\cdot0.5 = 0.2$, so **gain $= 0.32-0.2 = 0.12$**.
The tree picks whichever (feature, threshold) maximizes that gain.

## 4. Regression trees

Everything above predicts a **class**. The *same* tree machinery also predicts a
**number** — a price, a temperature, a demand — and then it's called a **regression
tree**. Only two pieces change.

**(a) What a leaf says.** A classification leaf outputs the *majority class* of its
training points; a regression leaf outputs their **average $y$**. So once the yes/no
questions route a new point down to a leaf, the prediction is simply the **mean target
of the training points that landed in that same box**. (Every point in a box gets the
*same* number — that's the key.)

**(b) How splits are chosen.** Gini and entropy measure *class* mixing, which is
meaningless for a number. Instead we measure **spread** with the variance (mean squared
error) of the $y$'s in a node, and pick the split that **shrinks it most**:
$$ \text{MSE(node)} = \tfrac1n\sum_i (y_i-\bar y)^2,\qquad \text{reduction} = \text{MSE(parent)} - \tfrac{n_L}{n}\text{MSE}(L) - \tfrac{n_R}{n}\text{MSE}(R). $$
A good split groups similar $y$'s together, so each child is "tighter" (lower variance)
than the parent — the regression analogue of making a node purer.

Because every leaf returns **one constant number**, the prediction as you sweep $x$ is a
**piecewise-constant staircase**: flat inside each box, jumping at each split. A
**shallow** tree makes a few wide steps (a coarse summary); a **deep** tree makes many
narrow steps that trace every wiggle — including the noise (overfitting, §5):

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 480 300" style="width:100%;max-width:470px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A regression tree predicts a piecewise-constant staircase. A shallow depth-2 tree makes a few wide steps; a deep depth-6 tree makes many narrow steps that chase the noisy data points."><rect x="1" y="1" width="478" height="298" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="45" y1="255" x2="448" y2="255" stroke="#C9C8C1" stroke-width="1.2"/><line x1="45" y1="50" x2="45" y2="255" stroke="#C9C8C1" stroke-width="1.2"/><polyline points="50,174 62,174 63,140 68,140 69,139 72,139 72,147 73,147 74,136 107,136 108,91 140,91 141,94 148,94 149,71 158,71 159,87 201,87 202,92 205,92 206,137 214,137 215,116 231,116 232,163 246,163 247,157 258,157 259,174 285,174 286,234 309,234 310,212 314,212 315,213 339,213 340,182 364,182 365,212 372,212 373,172 384,172 385,180 396,180 397,137 399,137 400,149 415,149 416,80 434,80 435,92 440,92" fill="none" stroke="#C0507A" stroke-width="1.5"/><polyline points="50,146 107,146 108,97 231,97 232,188 396,188 397,115 440,115" fill="none" stroke="#9A6A2A" stroke-width="2.6"/><g fill="#185FA5"><circle cx="57" cy="174" r="3.6"/><circle cx="68" cy="140" r="3.6"/><circle cx="69" cy="139" r="3.6"/><circle cx="71" cy="140" r="3.6"/><circle cx="74" cy="147" r="3.6"/><circle cx="75" cy="136" r="3.6"/><circle cx="139" cy="91" r="3.6"/><circle cx="141" cy="94" r="3.6"/><circle cx="156" cy="71" r="3.6"/><circle cx="161" cy="91" r="3.6"/><circle cx="200" cy="82" r="3.6"/><circle cx="203" cy="92" r="3.6"/><circle cx="209" cy="137" r="3.6"/><circle cx="220" cy="116" r="3.6"/><circle cx="242" cy="163" r="3.6"/><circle cx="251" cy="157" r="3.6"/><circle cx="267" cy="174" r="3.6"/><circle cx="304" cy="234" r="3.6"/><circle cx="314" cy="212" r="3.6"/><circle cx="315" cy="213" r="3.6"/><circle cx="364" cy="182" r="3.6"/><circle cx="365" cy="212" r="3.6"/><circle cx="379" cy="172" r="3.6"/><circle cx="389" cy="187" r="3.6"/><circle cx="390" cy="175" r="3.6"/><circle cx="393" cy="178" r="3.6"/><circle cx="399" cy="137" r="3.6"/><circle cx="400" cy="149" r="3.6"/><circle cx="430" cy="80" r="3.6"/><circle cx="440" cy="92" r="3.6"/></g><g font-family="sans-serif" font-size="11.5"><circle cx="230" cy="36" r="4" fill="#185FA5"/><text x="238" y="40" fill="#0C447C">data</text><line x1="284" y1="36" x2="306" y2="36" stroke="#9A6A2A" stroke-width="2.6"/><text x="310" y="40" fill="#9A6A2A">depth 2</text><line x1="372" y1="36" x2="394" y2="36" stroke="#C0507A" stroke-width="1.5"/><text x="398" y="40" fill="#C0507A">depth 6</text></g><text x="441" y="272" font-family="sans-serif" font-size="12" fill="#9C9B95">x</text><text x="32" y="54" font-family="sans-serif" font-size="12" fill="#9C9B95">y</text></svg></div>

**A one-line example.** A node holds four houses priced `[200k, 210k, 400k, 420k]`; their
mean (307.5k) is a poor single guess. Split on *"size &lt; 100 m²"* into `[200k, 210k]`
and `[400k, 420k]`: the children now predict ~205k and ~410k — far tighter, so this split
wins and the tree keeps going.

**The big limitation — no extrapolation.** A leaf only ever returns an *average of
training $y$'s*, so predictions are **boxed into the training range**. Give the tree an
$x$ beyond anything it saw and it just repeats the last step's value — it **cannot
continue a trend** the way a line $\hat y = wx+b$ does (notice the flat steps at both ends
of the staircase). Rule of thumb: trends that run off the edge of the data → a linear
model; flexible local structure → trees (and their ensembles, §7–§9).

## 5. Stopping & pruning (controlling overfitting)

A tree grown until every leaf is pure **memorizes the training set, noise and all** —
classic overfitting (deep tree ⇒ high variance). Two cures:
- **Pre-pruning** — cap growth with `max_depth`, `min_samples_split`,
  `min_samples_leaf`, `max_leaf_nodes`.
- **Post-pruning** — grow fully, then cut back branches that don't improve a penalized
  objective: **cost-complexity pruning** minimizes $\text{error} + \alpha\,(\#\text{leaves})$;
  larger $\alpha$ ⇒ smaller tree. Pick $\alpha$ by cross-validation (M6).

In the **Playground**, raise the depth and watch train accuracy march to ~100% while
test accuracy peaks then drops — the M0 U-curve, live.

## 6. Strengths & weaknesses

**Strengths:** non-linear boundaries; handles numeric **and** categorical features;
**no feature scaling** needed; invariant to monotone transforms; robust to outliers in
$x$; interpretable; fast to predict.
**Weaknesses:** a single tree is **unstable** (nudge the data → a very different tree)
and **high-variance**; splits are **axis-aligned only** (a diagonal boundary becomes a
staircase); greedy splitting can miss feature interactions; easy to overfit. The cure
for the variance is **ensembles**.

## 7. Why ensembles work — the variance math

Average $B$ models, each with variance $\sigma^2$. If they were **independent**, the
average has variance $\sigma^2/B$ — shrinking toward 0 with more models. Real trees
trained on similar data are **correlated** by some $\rho$, and the averaged variance is
$$ \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2. $$
So adding trees ($B\uparrow$) kills the **second** term, but the **first** ($\rho\sigma^2$)
is a floor set by correlation. This single formula explains both ensemble methods:
**bagging/forests attack $\rho$ and the second term; boosting attacks bias instead.**

## 8. Bagging & Random Forests

**Bagging** (bootstrap aggregating): train each tree on a **bootstrap sample** (draw
$n$ rows *with replacement*), then **vote/average**. Each deep tree is low-bias /
high-variance; averaging many cancels the variance.

A **Random Forest** adds the key trick: at each split, consider only a **random subset
of features** (typically $\sqrt{p}$ for classification). This **decorrelates** the trees
(lowers $\rho$ in §7), so averaging helps far more. Bonuses: **out-of-bag (OOB)** error
(free validation from the ~37% of rows each tree didn't see) and **feature importances**.
Forests are a superb, low-tuning default for tabular data.

## 9. Boosting

Build trees **sequentially**, each correcting the *current* ensemble's mistakes.
**Gradient boosting**: at each round fit a small tree to the **negative gradient of the
loss** (for squared error, just the residuals $y-\hat y$), then add it scaled by a small
**learning rate** $\eta$ (shrinkage):
$$ F_{m}(x) = F_{m-1}(x) + \eta\, h_m(x). $$

Watch it converge: start from the flat mean $F_0$, then each added tree bends the
prediction a little closer to the data — coarse after 1 tree, accurate after 60:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 480 300" style="width:100%;max-width:470px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gradient boosting starts from the flat mean F0 and adds shallow trees one at a time; after 1 tree the fit is coarse, after 8 closer, after 60 it tracks the data well."><rect x="1" y="1" width="478" height="298" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="45" y1="255" x2="448" y2="255" stroke="#C9C8C1" stroke-width="1.2"/><line x1="45" y1="50" x2="45" y2="255" stroke="#C9C8C1" stroke-width="1.2"/><g fill="#185FA5"><circle cx="57" cy="174" r="3"/><circle cx="68" cy="140" r="3"/><circle cx="69" cy="139" r="3"/><circle cx="71" cy="140" r="3"/><circle cx="74" cy="147" r="3"/><circle cx="75" cy="136" r="3"/><circle cx="139" cy="91" r="3"/><circle cx="141" cy="94" r="3"/><circle cx="156" cy="71" r="3"/><circle cx="161" cy="91" r="3"/><circle cx="200" cy="82" r="3"/><circle cx="203" cy="92" r="3"/><circle cx="209" cy="137" r="3"/><circle cx="220" cy="116" r="3"/><circle cx="242" cy="163" r="3"/><circle cx="251" cy="157" r="3"/><circle cx="267" cy="174" r="3"/><circle cx="304" cy="234" r="3"/><circle cx="314" cy="212" r="3"/><circle cx="315" cy="213" r="3"/><circle cx="364" cy="182" r="3"/><circle cx="365" cy="212" r="3"/><circle cx="379" cy="172" r="3"/><circle cx="389" cy="187" r="3"/><circle cx="390" cy="175" r="3"/><circle cx="393" cy="178" r="3"/><circle cx="399" cy="137" r="3"/><circle cx="400" cy="149" r="3"/><circle cx="430" cy="80" r="3"/><circle cx="440" cy="92" r="3"/></g><line x1="50" y1="146" x2="440" y2="146" stroke="#B0AFA8" stroke-width="1.3" stroke-dasharray="5 3"/><polyline points="50,146 107,146 108,136 231,136 232,154 396,154 397,139 440,139" fill="none" stroke="#9FD3BE" stroke-width="2"/><polyline points="50,157 62,157 63,142 107,142 108,102 205,102 206,128 231,128 232,160 258,160 259,180 285,180 286,188 372,188 373,173 396,173 397,148 415,148 416,104 440,104" fill="none" stroke="#3DA17F" stroke-width="2"/><polyline points="50,172 62,172 63,140 72,140 72,144 73,144 74,136 107,136 108,92 148,92 149,74 158,74 159,88 201,88 202,92 205,92 206,132 214,132 215,121 231,121 232,158 258,158 259,178 285,178 286,229 309,229 310,213 339,213 340,187 364,187 365,208 372,208 373,175 384,175 385,184 389,184 390,176 396,176 397,141 399,141 400,147 415,147 416,83 434,83 435,91 440,91" fill="none" stroke="#0E5E45" stroke-width="2.4"/><g font-family="sans-serif" font-size="11"><line x1="40" y1="34" x2="58" y2="34" stroke="#9FD3BE" stroke-width="2.4"/><text x="62" y="38" fill="#3DA17F">1 tree</text><line x1="116" y1="34" x2="134" y2="34" stroke="#3DA17F" stroke-width="2.4"/><text x="138" y="38" fill="#3DA17F">8 trees</text><line x1="206" y1="34" x2="224" y2="34" stroke="#0E5E45" stroke-width="2.4"/><text x="228" y="38" fill="#0E5E45">60 trees</text><line x1="320" y1="34" x2="338" y2="34" stroke="#B0AFA8" stroke-width="1.3" stroke-dasharray="5 3"/><text x="342" y="38" fill="#9C9B95">F₀ mean</text></g><text x="441" y="272" font-family="sans-serif" font-size="12" fill="#9C9B95">x</text><text x="32" y="54" font-family="sans-serif" font-size="12" fill="#9C9B95">y</text></svg></div>

Many shallow "weak" trees combine into a strong learner — this **reduces bias** and is,
on most **tabular** problems, the state of the art (**XGBoost, LightGBM, CatBoost**).
Regularize with small $\eta$ (+ more rounds), shallow trees (depth 3–8), row/column
**subsampling**, and L1/L2 penalties.

## 10. Bagging vs. boosting (the key contrast)

| | bagging / random forest | boosting |
|---|---|---|
| trees built | in **parallel**, independent | **sequentially**, each fixes errors |
| base learner | deep trees (low bias) | shallow stumps (weak) |
| mainly reduces | **variance** | **bias** |
| overfitting risk | low (more trees rarely hurt) | higher — tune $\eta$ / #rounds / depth |
| tuning effort | little | more |

## 11. Hyperparameters & feature importance

**Random forest:** `n_estimators` (more is better, diminishing returns), `max_features`
($\sqrt{p}$), `max_depth`/`min_samples_leaf` (light pruning). **Gradient boosting:**
`n_estimators` × `learning_rate` (trade them off), `max_depth` (3–8), `subsample`,
regularization. **Feature importance:** impurity-based importance is **biased** toward
high-cardinality / continuous features — prefer **permutation importance** (shuffle a
feature, measure the score drop) or SHAP values for trustworthy attributions.

## 12. Where trees shine — and where they don't

Gradient-boosted trees usually **win on tabular / structured data** (finance, health
records, most Kaggle tabular comps). Neural nets dominate **unstructured** data (images,
text, audio) where features must be *learned*. Good instinct: **tabular → start with
gradient boosting or a random forest; images/text → deep learning.**

## 13. Intuition recap

One tree is a flexible but jittery learner. **Bag** many to calm the jitter (variance),
or **boost** many weak ones to sharpen accuracy (bias). The ensemble — not the single
tree — is what wins.
"""

_M3_TASKS = r"""
### Warm-up — in the Playground tab
1. Set **max depth = 1** — describe the (under-fit) boundary on XOR-style data. Raise to
   2, then 8. Where does **test** accuracy peak, and where does train accuracy keep rising?
2. Increase the **label noise** — how does the best depth change?

### Pencil & paper
3. Compute the **Gini impurity** of a node with class counts [8, 2].
4. For a split that sends [8,2] → [6,0] and [2,2], compute the **information gain** (Gini).
5. Explain in one line why a single deep tree has high variance.

### Code
6. Implement Gini and a one-level "decision stump" (best single split) in NumPy.
7. Implement bagging: train 20 stumps on bootstrap samples, majority-vote, and compare
   accuracy/variance to a single tree.
8. Try `sklearn` `DecisionTreeClassifier`, `RandomForestClassifier`,
   `GradientBoostingClassifier` on the same data; compare.

### Bridge
9. Compare the tree's staircase boundary with the **linear** boundary from M2 logistic
   regression on the same XOR data — why does the linear model fail here?
"""

_M3_REFS = r"""
### Books
- James, Witten, Hastie & Tibshirani — *ISL*, ch. 8 (trees, bagging, boosting) — [free PDF](https://www.statlearning.com/).
- Hastie, Tibshirani & Friedman — *ESL*, ch. 9–10 & 15 (CART, boosting, random forests).

### Papers & docs
- Breiman (2001) — *Random Forests*. · Chen & Guestrin (2016) — *XGBoost*.
- scikit-learn — [trees](https://scikit-learn.org/stable/modules/tree.html) & [ensembles](https://scikit-learn.org/stable/modules/ensemble.html).

### In this lab
- ML: M0 (over/underfitting, bias–variance), M2 (the linear classifier trees beat on XOR).
"""

_M3_QUIZ = [
    Question("Each internal node of a standard decision tree splits on…",
             ["a linear combination of all features", "a single feature vs. a threshold (axis-aligned)",
              "the distance to a centroid", "a random label"], 1,
             "CART uses axis-aligned splits xⱼ ≤ t, carving space into rectangles."),
    Question("A tree chooses each split to…",
             ["maximize impurity", "maximize the reduction in impurity (information gain)",
              "minimize the number of nodes", "balance the classes exactly"], 1,
             "It greedily picks the split with the largest impurity reduction (Gini/entropy)."),
    Question("A very deep, unpruned decision tree tends to…",
             ["underfit (high bias)", "overfit (high variance) — it memorizes noise",
              "be perfectly calibrated", "ignore the data"], 1,
             "Without depth/leaf limits or pruning a tree fits training noise → high variance."),
    Question("Random forests reduce error mainly by…",
             ["reducing bias via sequential trees", "reducing variance by averaging many decorrelated trees",
              "scaling features", "using a single deep tree"], 1,
             "Bagging + random feature subsets decorrelate trees; averaging cuts variance."),
    Question("Gradient boosting differs from bagging because it…",
             ["trains trees in parallel", "builds trees sequentially, each correcting the previous errors (reduces bias)",
              "needs feature scaling", "uses only one tree"], 1,
             "Boosting is sequential and bias-reducing; bagging is parallel and variance-reducing."),
    Question("For typical tabular data, the usual state-of-the-art is…",
             ["a single decision tree", "gradient-boosted trees (XGBoost/LightGBM)",
              "a deep CNN", "k-means"], 1,
             "Boosted trees usually win on structured/tabular data; deep nets win on images/text."),
    Question("A practical advantage of trees over logistic regression / SVM is…",
             ["they output calibrated probabilities", "they need no feature scaling and handle non-linear, mixed-type data",
              "they are always more accurate", "they can't overfit"], 1,
             "Trees are scale-invariant and handle non-linearities and mixed feature types out of the box."),
    Question("In the ensemble variance formula ρσ² + (1−ρ)σ²/B, adding more trees (B→∞)…",
             ["removes all variance", "drives the second term to 0 but leaves the ρσ² floor",
              "increases bias", "has no effect"], 1,
             "More trees kill the averaged term; the correlation floor ρσ² is why forests also decorrelate (random features)."),
]

TREES = Lesson("trees", "Trees & ensembles", _M3_THEORY, _M3_QUIZ, _M3_TASKS, _M3_REFS)


# ===========================================================================
# M4 — SVM & kernels
# ===========================================================================

_M4_THEORY = r"""
## 1. The idea — the widest street

When two classes are cleanly separable, **many** different straight lines split them — so
*which* line is best? A **Support Vector Machine (SVM)** gives a crisp answer: pick the
boundary that leaves the **widest empty "street"** between the classes. Formally, among
all separating lines it **maximizes the margin** — the distance from the boundary out to
the nearest point on either side.

Why the widest? A boundary jammed up against the training points is fragile — a little
noise can push a point across it. A boundary parked in the **middle of the widest gap**
has the most breathing room left over, so it tends to **generalize best** to unseen data.
Think of steering down the **centre of the road** instead of hugging one curb.

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 460 270" style="width:100%;max-width:460px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Several thin lines can separate the two classes, but the SVM chooses the boundary sitting in the middle of the widest empty street; the two points touching the street edges are the support vectors."><defs><marker id="wsar" markerWidth="8" markerHeight="8" refX="4" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#6B6A66"/></marker></defs><rect x="1" y="1" width="458" height="268" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><rect x="150" y="28" width="100" height="222" fill="#EAF1F8"/><line x1="134" y1="28" x2="158" y2="250" stroke="#CBC9C2" stroke-width="1.4"/><line x1="264" y1="28" x2="242" y2="250" stroke="#CBC9C2" stroke-width="1.4"/><line x1="150" y1="28" x2="150" y2="250" stroke="#9C9B95" stroke-width="1.3" stroke-dasharray="5 3"/><line x1="250" y1="28" x2="250" y2="250" stroke="#9C9B95" stroke-width="1.3" stroke-dasharray="5 3"/><line x1="200" y1="28" x2="200" y2="250" stroke="#33312E" stroke-width="2.4"/><line x1="152" y1="208" x2="248" y2="208" stroke="#6B6A66" stroke-width="1.3" marker-start="url(#wsar)" marker-end="url(#wsar)"/><text x="200" y="201" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#6B6A66">margin</text><g fill="#A32D2D"><circle cx="58" cy="170" r="4.5"/><circle cx="90" cy="210" r="4.5"/><circle cx="110" cy="242" r="4.5"/><circle cx="78" cy="128" r="4.5"/><circle cx="150" cy="120" r="4.5"/></g><g fill="#185FA5"><circle cx="300" cy="92" r="4.5"/><circle cx="332" cy="132" r="4.5"/><circle cx="356" cy="166" r="4.5"/><circle cx="304" cy="212" r="4.5"/><circle cx="250" cy="160" r="4.5"/></g><g fill="none" stroke="#33312E" stroke-width="1.6"><circle cx="150" cy="120" r="9"/><circle cx="250" cy="160" r="9"/></g><text x="200" y="20" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#9C9B95">thin grey lines also separate — but hug the points</text><text x="62" y="262" font-family="sans-serif" font-size="11" fill="#33312E">support vectors (circled) touch the street edges</text></svg></div>

The two cluster-edge points the street just touches are special — they alone pin the
boundary in place (that's §3). Everything below builds this up precisely, then teaches it
to tolerate overlap (§4) and bend into curves (§6).

## 2. The margin, made precise

Put the boundary at $\mathbf w\cdot\mathbf x + b = 0$ and scale $(\mathbf w,b)$ so the
closest points satisfy $y_i(\mathbf w\cdot\mathbf x_i+b) = 1$. Then the **street's width**
is $\dfrac{2}{\lVert\mathbf w\rVert}$. Maximizing the margin is therefore

$$ \min_{\mathbf w,b}\ \tfrac12\lVert\mathbf w\rVert^2 \quad\text{s.t.}\quad y_i(\mathbf w\cdot\mathbf x_i+b)\ge 1\ \forall i. $$

A clean convex (quadratic) program with a unique solution.

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 420 300" style="width:100%;max-width:420px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A support vector machine separates two classes with the widest possible street: a solid boundary line, two dashed margin lines, and the circled points touching them are the support vectors."><defs><marker id="svar" markerWidth="8" markerHeight="8" refX="4" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#6B6A66"/></marker></defs><rect x="1" y="1" width="418" height="298" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="45" y1="250" x2="325" y2="80" stroke="#9C9B95" stroke-width="1.5" stroke-dasharray="6 4"/><line x1="115" y1="250" x2="395" y2="80" stroke="#9C9B95" stroke-width="1.5" stroke-dasharray="6 4"/><line x1="80" y1="250" x2="360" y2="80" stroke="#33312E" stroke-width="2.4"/><line x1="187" y1="165" x2="253" y2="165" stroke="#6B6A66" stroke-width="1.4" marker-start="url(#svar)" marker-end="url(#svar)"/><text x="220" y="157" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#6B6A66">margin</text><g fill="#A32D2D"><circle cx="70" cy="205" r="4.5"/><circle cx="60" cy="240" r="4.5"/><circle cx="110" cy="258" r="4.5"/><circle cx="48" cy="180" r="4.5"/><circle cx="185" cy="165" r="4.5"/><circle cx="103" cy="215" r="4.5"/></g><g fill="#185FA5"><circle cx="360" cy="95" r="4.5"/><circle cx="385" cy="135" r="4.5"/><circle cx="330" cy="72" r="4.5"/><circle cx="300" cy="150" r="4.5"/><circle cx="255" cy="165" r="4.5"/><circle cx="341" cy="113" r="4.5"/></g><g fill="none" stroke="#33312E" stroke-width="1.6"><circle cx="185" cy="165" r="9"/><circle cx="103" cy="215" r="9"/><circle cx="255" cy="165" r="9"/><circle cx="341" cy="113" r="9"/></g><text x="120" y="115" font-family="sans-serif" font-size="11" fill="#33312E">support vectors (circled)</text><text x="305" y="215" font-family="sans-serif" font-size="12" fill="#185FA5">class +1</text><text x="92" y="288" font-family="sans-serif" font-size="12" fill="#A32D2D">class −1</text></svg></div>

## 3. Support vectors

Here is the SVM's most surprising property. At the optimum, **only the points sitting
right on the street's edges matter** — the ones with $y_i(\mathbf w\cdot\mathbf x_i+b)=1$.
These are the **support vectors**: they "support" (hold up) the boundary the way tent
poles hold up a tent. Every *other* point — anything sitting comfortably inside its own
class — could be **moved, or even deleted, and the boundary would not budge** at all.

Three consequences follow:

- **Sparsity** — the trained model is defined by just a handful of points, not the whole
  dataset. A prediction only compares the new point against those few.
- **Robustness** — the bulk of the data and any far-away points have *zero* influence, so
  the fit is decided by the **hard, borderline** cases, never swayed by easy ones.
- **A caution** — because a few border points decide everything, one mislabeled point
  right on the margin can shift the boundary noticeably. The soft margin (§4) is what
  guards against that.

## 4. Soft margin & the C knob

Real data **overlaps** — usually no straight line separates it perfectly. A strict "hard
margin" that demands zero mistakes would then be impossible, or would contort itself
around a single outlier. The fix is the **soft margin**: give each point some **slack**
$\xi_i\ge 0$ — the amount it's allowed to intrude into the street, or even sit on the
wrong side — and then *charge* for that slack:
$$ \min\ \tfrac12\lVert\mathbf w\rVert^2 + C\sum_i\xi_i \quad\text{s.t.}\quad y_i(\mathbf w\cdot\mathbf x_i+b)\ge 1-\xi_i. $$
The first term still wants a **wide** street; the second adds a penalty $C$ for every unit
of violation. So **C is the regularization dial** — precisely the M0 bias–variance knob:

- **Large C** → violations are expensive → the street **narrows** to avoid them → fits the
  training data tightly (**low bias, high variance** → can overfit).
- **Small C** → violations are cheap → the street **widens**, shrugging off a few
  stragglers → a smoother, more general boundary (**high bias, low variance**).

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 520 250" style="width:100%;max-width:520px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two panels. Small C gives a wide street that tolerates a few points violating the margin. Large C gives a narrow street that hugs the points to avoid violations."><rect x="1" y="1" width="518" height="248" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><rect x="95" y="35" width="95" height="170" fill="#EAF1F8"/><line x1="95" y1="35" x2="95" y2="205" stroke="#9C9B95" stroke-width="1.2" stroke-dasharray="5 3"/><line x1="190" y1="35" x2="190" y2="205" stroke="#9C9B95" stroke-width="1.2" stroke-dasharray="5 3"/><line x1="142" y1="35" x2="142" y2="205" stroke="#33312E" stroke-width="2.2"/><g fill="#A32D2D"><circle cx="40" cy="90" r="4.5"/><circle cx="58" cy="150" r="4.5"/><circle cx="72" cy="192" r="4.5"/><circle cx="48" cy="118" r="4.5"/><circle cx="156" cy="104" r="4.5"/></g><g fill="#185FA5"><circle cx="220" cy="80" r="4.5"/><circle cx="234" cy="140" r="4.5"/><circle cx="246" cy="186" r="4.5"/><circle cx="210" cy="110" r="4.5"/><circle cx="120" cy="176" r="4.5"/></g><text x="142" y="226" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#33312E">small C — wide street, tolerates violations</text><rect x="375" y="35" width="34" height="170" fill="#EAF1F8"/><line x1="375" y1="35" x2="375" y2="205" stroke="#9C9B95" stroke-width="1.2" stroke-dasharray="5 3"/><line x1="409" y1="35" x2="409" y2="205" stroke="#9C9B95" stroke-width="1.2" stroke-dasharray="5 3"/><line x1="392" y1="35" x2="392" y2="205" stroke="#33312E" stroke-width="2.2"/><g fill="#A32D2D"><circle cx="300" cy="92" r="4.5"/><circle cx="322" cy="150" r="4.5"/><circle cx="340" cy="192" r="4.5"/><circle cx="356" cy="120" r="4.5"/><circle cx="374" cy="150" r="4.5"/></g><g fill="#185FA5"><circle cx="478" cy="80" r="4.5"/><circle cx="492" cy="140" r="4.5"/><circle cx="458" cy="186" r="4.5"/><circle cx="436" cy="110" r="4.5"/><circle cx="410" cy="134" r="4.5"/></g><text x="430" y="226" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#33312E">large C — narrow street, hugs the points</text></svg></div>

You choose C by cross-validation (§8, M6); the best value depends on how noisy and
overlapped your data is.

## 5. The hinge-loss view

The soft-margin program has a second, very useful face: it is exactly the same as
minimizing the **hinge loss** plus an L2 penalty:
$$ \sum_i \max\!\big(0,\ 1 - y_i\,f(\mathbf x_i)\big) + \lambda\lVert\mathbf w\rVert^2,\qquad \lambda = \tfrac{1}{2C}. $$
Read it through the **margin** $m = y_i\,f(\mathbf x_i)$ — positive when a point is on the
correct side, larger the further it is from the boundary:

- $m \ge 1$ (correct **and** past the street) → loss is **exactly 0**: the point is happy
  and contributes nothing to the fit.
- $0 \le m < 1$ (correct but inside the street) → a small, growing penalty.
- $m < 0$ (wrong side) → the penalty keeps growing linearly.

So only **border and violating** points push the boundary — which is the loss-function
reason the answer depends on the support vectors alone (§3). Here are three losses on one
axis:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 460 280" style="width:100%;max-width:460px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Three classification losses versus the margin: the 0/1 loss is a step at zero, the hinge loss falls linearly and becomes exactly zero once the margin passes 1 (with a kink there), and the log loss is smooth and never reaches zero."><rect x="1" y="1" width="458" height="278" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="55" y1="235" x2="440" y2="235" stroke="#C9C8C1" stroke-width="1.2"/><line x1="55" y1="38" x2="55" y2="235" stroke="#C9C8C1" stroke-width="1.2"/><line x1="205" y1="42" x2="205" y2="235" stroke="#D9D8D1" stroke-width="1" stroke-dasharray="4 3"/><line x1="280" y1="42" x2="280" y2="235" stroke="#ECEBE4" stroke-width="1" stroke-dasharray="4 3"/><polyline points="55,172 205,172 205,235 440,235" fill="none" stroke="#9C9B95" stroke-width="1.6" stroke-dasharray="5 3"/><polyline points="55,45 61,47 68,54 74,61 80,67 87,74 93,80 99,86 106,93 112,99 119,105 125,110 131,116 138,122 144,127 150,132 157,138 163,143 169,147 176,152 182,157 188,161 195,165 201,169 208,173 214,177 220,180 227,184 233,187 239,190 246,193 252,196 258,199 265,201 271,203 277,206 284,208 290,210 297,211 303,213 309,215 316,216 322,218 328,219 335,220 341,221 347,222 354,223 360,224 366,225 373,226 379,226 386,227 392,228 398,228 405,229 411,229 417,230 424,230 430,231" fill="none" stroke="#185FA5" stroke-width="2.4"/><polyline points="55,45 87,72 119,99 150,126 182,152 214,179 246,206 277,233 284,235 430,235" fill="none" stroke="#9A6A2A" stroke-width="2.4"/><g font-family="sans-serif" font-size="11" fill="#6B6A66"><text x="205" y="250" text-anchor="middle">m=0</text><text x="280" y="250" text-anchor="middle">m=1</text><text x="392" y="252">margin m = y·f(x)</text><text x="36" y="50">loss</text></g><g font-family="sans-serif" font-size="11.5"><line x1="320" y1="58" x2="342" y2="58" stroke="#9A6A2A" stroke-width="2.4"/><text x="346" y="62" fill="#9A6A2A">hinge</text><line x1="320" y1="76" x2="342" y2="76" stroke="#185FA5" stroke-width="2.4"/><text x="346" y="80" fill="#185FA5">log loss</text><line x1="320" y1="94" x2="342" y2="94" stroke="#9C9B95" stroke-width="1.6" stroke-dasharray="5 3"/><text x="346" y="98" fill="#9C9B95">0/1 loss</text></g></svg></div>

The dashed **0/1 loss** is what we'd *love* to minimize (just the count of mistakes), but
it's flat with a cliff — non-differentiable, useless for optimization. **Hinge** is its
convex stand-in with the tell-tale **kink at $m=1$** (the SVM's choice). **Log loss**
(logistic regression, M2) is smooth and **never quite reaches zero**, so *every* point
keeps nudging the boundary — which is why logistic regression yields calibrated
probabilities, while the SVM yields a sparse, margin-focused boundary. Same linear model,
**different loss curve**.

## 6. The kernel trick — non-linear boundaries for free

A straight line is useless when the classes curl around each other. The classic escape is
to **add features**. Picture 1-D data where one class sits in the **middle** and the other
at **both ends** (left below): no single threshold can split them. But **lift** each point
with an extra coordinate $x^2$ and they rearrange in 2-D so that a *straight* line
separates them (right) — and that straight line corresponds to a **curved** boundary back
in the original 1-D space:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 520 280" style="width:100%;max-width:520px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Left: in one dimension the blue class is in the middle and red on both ends, so no single threshold separates them. Right: adding an x-squared coordinate lifts the points onto a parabola, where one straight horizontal line cleanly separates blue (low) from red (high)."><rect x="1" y="1" width="518" height="278" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><text x="130" y="42" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#33312E">original 1-D — no cut works</text><line x1="30" y1="150" x2="230" y2="150" stroke="#C9C8C1" stroke-width="1.4"/><g fill="#185FA5"><circle cx="100" cy="150" r="5"/><circle cx="120" cy="150" r="5"/><circle cx="145" cy="150" r="5"/><circle cx="160" cy="150" r="5"/></g><g fill="#A32D2D"><circle cx="40" cy="150" r="5"/><circle cx="65" cy="150" r="5"/><circle cx="190" cy="150" r="5"/><circle cx="215" cy="150" r="5"/></g><text x="130" y="180" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#9C9B95">red · blue · red  along x</text><text x="252" y="146" font-family="sans-serif" font-size="22" fill="#6B6A66">→</text><text x="256" y="128" font-family="sans-serif" font-size="10" fill="#6B6A66">add x²</text><text x="396" y="42" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#33312E">lifted to 2-D (x, x²)</text><line x1="290" y1="255" x2="505" y2="255" stroke="#C9C8C1" stroke-width="1.2"/><line x1="290" y1="40" x2="290" y2="255" stroke="#C9C8C1" stroke-width="1.2"/><polyline points="290,45 295,65 300,84 306,102 311,119 316,135 322,150 327,163 332,176 337,188 342,199 348,208 353,217 358,225 364,232 369,237 374,242 379,245 384,248 390,249 395,250 400,249 406,248 411,245 416,242 421,237 426,232 432,225 437,217 442,208 448,199 453,188 458,176 463,163 468,150 474,135 479,119 484,102 490,84 495,65 500,45" fill="none" stroke="#D9D8D1" stroke-width="1.6"/><line x1="293" y1="199" x2="503" y2="199" stroke="#1D9E75" stroke-width="2" stroke-dasharray="6 3"/><text x="500" y="193" text-anchor="end" font-family="sans-serif" font-size="10" fill="#1D9E75">one straight line!</text><g fill="#185FA5"><circle cx="364" cy="232" r="5"/><circle cx="384" cy="248" r="5"/><circle cx="411" cy="245" r="5"/><circle cx="426" cy="232" r="5"/></g><g fill="#A32D2D"><circle cx="300" cy="84" r="5"/><circle cx="327" cy="163" r="5"/><circle cx="458" cy="176" r="5"/><circle cx="484" cy="102" r="5"/></g><text x="283" y="50" font-family="sans-serif" font-size="10" fill="#9C9B95">x²</text></svg></div>

That's the whole idea: map inputs through some $\phi(\mathbf x)$ into a higher-dimensional
space where the classes **are** linearly separable, then run the ordinary max-margin SVM
*there*. The snag: $\phi$ can be enormous — even **infinite-dimensional** — so building it
explicitly is hopeless.

**The trick.** The SVM's optimization (in its dual form) touches the data **only through
dot products** $\mathbf x_i\cdot\mathbf x_j$ — never the points on their own. A **kernel**
$K(\mathbf x_i,\mathbf x_j)=\phi(\mathbf x_i)\cdot\phi(\mathbf x_j)$ returns that dot
product in the lifted space **directly from the original vectors**, without ever
constructing $\phi$. So you get the power of a giant feature space for the price of a tiny
function — a non-linear boundary essentially "for free". Any $K$ that corresponds to a
valid inner product (**Mercer's condition**) is allowed.

## 7. Common kernels

- **Linear** $K=\mathbf x\cdot\mathbf x'$ — the plain max-margin line; great for
  high-dimensional/sparse data (text/TF-IDF).
- **Polynomial** $K=(\mathbf x\cdot\mathbf x' + c)^d$ — degree-$d$ curved boundaries.
- **RBF / Gaussian** $K=e^{-\gamma\lVert\mathbf x-\mathbf x'\rVert^2}$ — the workhorse.
  It's a **similarity** that decays with distance, mapping to an *infinite*-dimensional
  space. **gamma** sets each point's reach: large γ → tight, wiggly boundary (overfit);
  small γ → smooth, almost linear.

## 8. Choosing C and γ (and scaling!)

An RBF SVM has **two** dials, and they interact:

- **C** (§4) — how hard to punish margin violations: the bias–variance knob (large C →
  tight fit; small C → smoother).
- **γ** (gamma, §7) — how far each point's influence reaches. **Large γ** = short reach =
  the boundary wiggles tightly around individual points (**overfit**); **small γ** = long
  reach = a smooth, nearly linear boundary (**underfit**).

Because the two trade off, tune them **together** with a cross-validated **grid search**
(M6) over **log-spaced** values, e.g. C, γ ∈ {0.01, 0.1, 1, 10, 100}, keeping the pair
with the best validation score.

**Always standardize the features first.** The RBF kernel measures Euclidean distance
$\lVert\mathbf x-\mathbf x'\rVert$, so a feature on a bigger numeric scale (income in
dollars vs. age in years) silently dominates that distance and the others are effectively
ignored. Rescaling to comparable ranges (e.g. `StandardScaler`) fixes it — and this
scale-sensitivity is the **#1 SVM gotcha**. (Trees, by contrast, don't care about scale —
M3 §6.)

## 9. Multiclass & regression

**More than two classes.** An SVM is binary at heart (one street between two sides), so
libraries combine several:

- **One-vs-rest (OvR)** — train $k$ SVMs, each "class $c$ vs. all the rest", and predict
  the class whose boundary scores highest ($k$ models).
- **One-vs-one (OvO)** — train one SVM for **every pair** of classes and let them vote
  ($\binom{k}{2}$ models, each on less data; this is scikit-learn's default for `SVC`).

**Regression (SVR).** The same margin idea, turned inside out: instead of a street
*between* classes, fit a **tube of half-width $\varepsilon$ around the data**. Points
**inside** the tube cost nothing — an "$\varepsilon$-insensitive" zone where small errors
are ignored — and only points **outside** it are penalized. The result is the flattest
function that stays within $\varepsilon$ of most points: the regression cousin of "only
the border points matter".

## 10. Strengths & weaknesses

**Strengths:** effective in **high dimensions** (even features > samples), strong when a
clear margin exists, flexible via kernels, sparse/robust (only support vectors matter).
**Weaknesses:** training is $O(n^2)$–$O(n^3)$, so **slow on large datasets**;
**scale-sensitive**; no native probabilities (need calibration, M2 §12); kernel/C/γ
choices need tuning; less interpretable than a tree or linear model.

## 11. Where SVMs are used

Text classification (a linear SVM on TF-IDF is a classic strong baseline),
bioinformatics, handwritten-digit and image classification before deep learning, and
small-to-medium high-dimensional problems with a clean margin.

## 12. SVM vs. logistic regression vs. trees (when to pick what)

- **Clear margin, high-dim, medium data, want max-margin robustness →** SVM.
- **Want probabilities / interpretable coefficients / very large data →** logistic regression.
- **Tabular, non-linear, mixed types, minimal preprocessing →** trees / gradient boosting.

All three draw the *same* linear boundary in the linear case; the differences are the
loss, the kernel option, and the engineering tradeoffs.
"""

_M4_TASKS = r"""
### Pencil & paper
1. Sketch two separable clusters and draw the max-margin line; mark the support vectors.
2. Explain what changes if you move (a) a support vector, (b) a far-away point.
3. Describe what large C vs. small C does to the margin and to overfitting.

### Code
4. `sklearn.svm.SVC(kernel="linear")` vs `kernel="rbf"` on the M2 / XOR data; compare boundaries.
5. Sweep **C** and **gamma** (RBF) on a validation set; plot how the boundary changes.
6. Standardize the features first and show how much it matters for the RBF SVM.

### Concept
7. Why does the kernel trick let you get non-linear boundaries *without* computing the
   high-dimensional mapping? (Hint: the optimization only needs dot products.)

### Bridge
8. Compare an RBF-SVM boundary on XOR with the decision tree (M3) and logistic regression
   (M2). Which assumptions does each make?
"""

_M4_REFS = r"""
### Books
- James, Witten, Hastie & Tibshirani — *ISL*, ch. 9 (SVMs) — [free PDF](https://www.statlearning.com/).
- Hastie, Tibshirani & Friedman — *ESL*, ch. 12. · Bishop — *PRML*, ch. 7.

### Docs
- scikit-learn — [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html) (incl. the kernel & C/gamma guide).

### In this lab
- ML: M2 (linear classifier & hinge-vs-log loss), M0 (C as the bias–variance knob).
"""

_M4_QUIZ = [
    Question("An SVM chooses the separating boundary that…",
             ["passes through the most points", "maximizes the margin (distance to the nearest points)",
              "minimizes the number of features", "is always vertical"], 1,
             "SVMs are max-margin classifiers — the widest street between the classes."),
    Question("Support vectors are…",
             ["all the training points", "only the points closest to the boundary, which define it",
              "the features", "the test points"], 1,
             "Only the nearest points (support vectors) determine the boundary; others don't matter."),
    Question("The kernel trick lets an SVM…",
             ["train faster on huge data", "produce non-linear boundaries by using dot products in a high-dim space without computing the mapping",
              "avoid choosing C", "output probabilities"], 1,
             "Kernels compute high-dimensional dot products implicitly → non-linear boundaries cheaply."),
    Question("Increasing C in a soft-margin SVM…",
             ["widens the margin and regularizes more", "narrows the margin and fits the training data harder (less regularization)",
              "has no effect", "removes the support vectors"], 1,
             "Large C punishes margin violations → narrower margin, lower bias / higher variance."),
    Question("Before training an RBF SVM you should always…",
             ["shuffle the labels", "standardize/scale the features", "remove the bias term", "use accuracy only"], 1,
             "SVMs (especially RBF, which uses distances) are scale-sensitive — standardize features."),
    Question("Compared to logistic regression, a linear SVM…",
             ["uses log loss and gives probabilities", "uses hinge loss and focuses on the margin (border points)",
              "cannot separate classes", "needs no boundary"], 1,
             "Hinge loss ignores points correctly classified beyond the margin; log loss uses all points + gives probabilities."),
    Question("Maximizing the SVM margin is equivalent to…",
             ["maximizing ‖w‖", "minimizing ½‖w‖² subject to the points being correctly classified by ≥1",
              "minimizing the number of support vectors", "maximizing C"], 1,
             "Margin width = 2/‖w‖, so maximizing it means minimizing ½‖w‖² under the margin constraints."),
]

SVM = Lesson("svm", "SVM & kernels", _M4_THEORY, _M4_QUIZ, _M4_TASKS, _M4_REFS)


# ===========================================================================
# M5 — Unsupervised learning
# ===========================================================================

_M5_THEORY = r"""
## 1. What unsupervised learning is

**Supervised** learning (M1–M4) trains on labeled pairs $(\mathbf x, y)$ — you hand it the
right answers. **Unsupervised** learning gets only the inputs $\mathbf x$, with **no labels
at all**, and must discover structure on its own. Three main jobs:

- **Clustering** — group similar points together (customer segments, topics in documents).
- **Dimensionality reduction** — squeeze many features into a few that keep most of the
  information (for visualization, speed, or denoising).
- **Anomaly detection** — flag the points that *don't* fit the structure (fraud, faults).

The catch: with no ground-truth labels there is **no "accuracy" to optimize**, so deciding
whether the structure you found is any *good* — and even **how many clusters exist** —
becomes the hard part (§13). It's also how much of modern AI learns: the world has far more
unlabeled than labeled data, so finding structure without labels is enormously valuable (§14).

## 2. k-means — objective & algorithm

Partition points into **k** clusters, each summarized by a **centroid** $\mu_j$.
k-means minimizes the **inertia** (within-cluster sum of squares):
$$ J = \sum_{i} \lVert \mathbf x_i - \mu_{c(i)}\rVert^2 . $$
**Lloyd's algorithm** is coordinate descent on $J$, alternating:
1. **Assign** each point to its nearest centroid (fix $\mu$, optimize assignments).
2. **Update** each centroid to the mean of its points (fix assignments, optimize $\mu$).

Each step can only lower $J$, so it **converges — but to a local minimum**, not
necessarily the best one.

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 360 300" style="width:100%;max-width:360px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="k-means clustering with k equals 3: points colored by their assigned cluster, with an X marking each cluster's centroid (mean)."><rect x="1" y="1" width="358" height="298" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g fill="#1D9E75"><circle cx="129" cy="182" r="4.5"/><circle cx="114" cy="208" r="4.5"/><circle cx="124" cy="193" r="4.5"/><circle cx="152" cy="181" r="4.5"/><circle cx="127" cy="176" r="4.5"/><circle cx="119" cy="206" r="4.5"/><circle cx="135" cy="183" r="4.5"/><circle cx="119" cy="204" r="4.5"/></g><g fill="#9A6A2A"><circle cx="253" cy="224" r="4.5"/><circle cx="262" cy="181" r="4.5"/><circle cx="215" cy="187" r="4.5"/><circle cx="228" cy="177" r="4.5"/><circle cx="207" cy="201" r="4.5"/><circle cx="262" cy="171" r="4.5"/><circle cx="204" cy="195" r="4.5"/><circle cx="255" cy="197" r="4.5"/></g><g fill="#7F77DD"><circle cx="222" cy="119" r="4.5"/><circle cx="197" cy="117" r="4.5"/><circle cx="219" cy="102" r="4.5"/><circle cx="182" cy="127" r="4.5"/><circle cx="224" cy="91" r="4.5"/><circle cx="205" cy="85" r="4.5"/><circle cx="197" cy="111" r="4.5"/><circle cx="174" cy="113" r="4.5"/></g><g stroke-width="3.5" stroke-linecap="round"><g stroke="#0E5E45"><line x1="119" y1="183" x2="135" y2="199"/><line x1="119" y1="199" x2="135" y2="183"/></g><g stroke="#5A3E14"><line x1="228" y1="184" x2="244" y2="200"/><line x1="228" y1="200" x2="244" y2="184"/></g><g stroke="#3F3A8C"><line x1="195" y1="100" x2="211" y2="116"/><line x1="195" y1="116" x2="211" y2="100"/></g></g><text x="180" y="285" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#9C9B95">dots = points colored by cluster   ·   ✕ = centroid (mean)</text></svg></div>

## 3. Initialization matters — k-means++ & restarts

Because Lloyd's algorithm only reaches a **local** minimum, *where you start* changes where
you end up. Seed two centroids inside the same true cluster and k-means can lock into a
clearly wrong grouping and never escape. Two standard fixes:

- **k-means++** — choose the starting centroids to be **spread out**: the first is random,
  and each next seed is picked with probability proportional to its **squared distance** from
  the nearest existing seed. This cheap trick avoids clumped starts and gives much better,
  more consistent results (it's scikit-learn's default).
- **Multiple restarts** — run the whole thing several times from different seeds and keep the
  solution with the **lowest inertia** (the `n_init` setting).

Together they make k-means reliable in practice. (Change the seed in the Playground and
watch the clustering shift.)

## 4. Choosing k

k-means needs **k fixed in advance**, but no label tells you the right number. The trap:
**inertia always falls as k rises** (more centroids ⇒ tighter clusters; at k = n it hits
0), so you *can't* just minimize it. Instead:

- **Elbow method** — plot inertia vs. k and look for the **"elbow"**, where adding another
  cluster stops buying much. Before the elbow each new cluster captures real structure;
  after it you're just splitting noise. Here the bend at **k = 3** matches the three true
  blobs:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 440 280" style="width:100%;max-width:440px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Elbow plot: k-means inertia drops steeply from k=1 to k=3 then flattens, so the elbow at k=3 is the suggested number of clusters."><rect x="1" y="1" width="438" height="278" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="55" y1="235" x2="425" y2="235" stroke="#C9C8C1" stroke-width="1.2"/><line x1="55" y1="40" x2="55" y2="235" stroke="#C9C8C1" stroke-width="1.2"/><line x1="160" y1="46" x2="160" y2="235" stroke="#1D9E75" stroke-width="1.3" stroke-dasharray="5 3"/><text x="160" y="38" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1D9E75">elbow → k=3</text><polyline points="60,50 110,205 160,223 210,224 260,225 310,226 360,226 410,227" fill="none" stroke="#185FA5" stroke-width="2.4"/><g fill="#185FA5"><circle cx="60" cy="50" r="3.5"/><circle cx="110" cy="205" r="3.5"/><circle cx="160" cy="223" r="3.5"/><circle cx="210" cy="224" r="3.5"/><circle cx="260" cy="225" r="3.5"/><circle cx="310" cy="226" r="3.5"/><circle cx="360" cy="226" r="3.5"/><circle cx="410" cy="227" r="3.5"/></g><g font-family="sans-serif" font-size="10" fill="#9C9B95" text-anchor="middle"><text x="60" y="250">1</text><text x="110" y="250">2</text><text x="160" y="250">3</text><text x="210" y="250">4</text><text x="260" y="250">5</text><text x="310" y="250">6</text><text x="360" y="250">7</text><text x="410" y="250">8</text></g><text x="240" y="268" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#6B6A66">k (number of clusters)</text><text x="34" y="50" font-family="sans-serif" font-size="11" fill="#6B6A66">inertia</text></svg></div>

- **Silhouette score** — for each point let $a$ = mean distance to its *own* cluster and
  $b$ = mean distance to the *nearest other* cluster; its silhouette is
  $s = \dfrac{b-a}{\max(a,b)} \in [-1,1]$ (near $+1$ = snug and well-separated; $0$ = on a
  border; negative = probably in the wrong cluster). Average over all points and pick the k
  with the highest mean — often more reliable than eyeballing the elbow.
- **Gap statistic**, or simplest of all, let a **downstream task** decide which k is useful.

## 5. k-means limitations

k-means is fast and simple, but its built-in assumptions bite often:

- **It expects round, equal-size blobs.** Inertia + Euclidean distance implicitly assume
  **spherical clusters of similar size and density**. Elongated, crescent, or nested-ring
  shapes get sliced the wrong way (you'll see exactly this in §7).
- **You must choose k up front** (§4) — awkward when you don't know it.
- **Scale-sensitive** — distances blend all features, so an unscaled large-range feature
  dominates. **Standardize first.**
- **Outlier-sensitive** — a single far-off point drags its centroid, because the mean
  isn't robust.
- **Hard, convex assignments only** — every point is 100% one cluster (no "60% A, 40% B"),
  and regions are always convex.

When these bite, the next sections give alternatives: **arbitrary shapes** (DBSCAN, §7),
**no need to fix k** (hierarchical §6, DBSCAN §7), and **soft memberships** (GMM, §8).

## 6. Hierarchical clustering

Instead of committing to one k, hierarchical clustering builds a **whole tree of nested
groupings** called a **dendrogram**. The common **agglomerative** ("bottom-up") version:

1. Start with every point as its own cluster.
2. Repeatedly **merge the two closest clusters**.
3. Continue until everything is one cluster — recording each merge and the **height**
   (distance) at which it happened.

"Closest" depends on the **linkage** rule: **single** (nearest pair — tends to make long
chains), **complete** (farthest pair — compact balls), **average**, or **Ward** (the merge
that least increases variance — behaves like k-means). The dendrogram's height is the
distance at which two clusters joined; **cut it at any height** and you read off that many
clusters — so you can choose k *after* seeing the structure, not before:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 440 280" style="width:100%;max-width:440px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A dendrogram of six points A to F merged bottom-up at increasing distances; a dashed horizontal cut line crosses three branches, giving three clusters: A-B-C, D-E, and F."><rect x="1" y="1" width="438" height="278" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g stroke="#33312E" stroke-width="1.8" fill="none"><path d="M60,230 V195 H120 V230"/><path d="M240,230 V190 H300 V230"/><path d="M90,195 V160 H180 V230"/><path d="M270,190 V108 H360 V230"/><path d="M135,160 V95 H315 V108"/></g><line x1="40" y1="125" x2="405" y2="125" stroke="#1D9E75" stroke-width="1.4" stroke-dasharray="6 3"/><text x="400" y="120" text-anchor="end" font-family="sans-serif" font-size="10.5" fill="#1D9E75">cut → 3 clusters</text><g font-family="sans-serif" font-size="12" fill="#33312E" text-anchor="middle"><text x="60" y="247">A</text><text x="120" y="247">B</text><text x="180" y="247">C</text><text x="240" y="247">D</text><text x="300" y="247">E</text><text x="360" y="247">F</text></g><text x="26" y="150" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#6B6A66" transform="rotate(-90 26 150)">distance →</text></svg></div>

Great for small datasets and for *seeing* nested structure; the catch is it costs about
$O(n^2)$ time and memory, so it doesn't scale to very large data.

## 7. Density-based clustering (DBSCAN)

DBSCAN takes a completely different view: a cluster is a **dense region** of points, set
off from others by **sparser gaps**. With two settings — a radius `eps` and a count
`min_samples` — it labels each point:

- **core** — has at least `min_samples` neighbors within `eps` (deep inside a dense region);
- **border** — within `eps` of a core point, but not dense itself;
- **noise** — neither, i.e. an **outlier** (DBSCAN happily leaves points unassigned).

Clusters then grow by chaining neighboring core points together. The payoffs are big: it
finds **arbitrarily shaped** clusters, **doesn't need k**, and **flags outliers** for
free. Below it cleanly separates two interleaved crescents that k-means — which can only
cut a **straight** boundary between its two centroids — gets completely wrong:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 520 280" style="width:100%;max-width:520px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two interleaving crescent clusters. Left: k-means with k=2 splits them with a straight cut and mixes the crescents. Right: DBSCAN follows the density and recovers the two crescents correctly."><rect x="1" y="1" width="518" height="278" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><text x="135" y="22" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#A32D2D">k-means (k=2) — straight cut, wrong</text><text x="392" y="22" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#1D9E75">DBSCAN — follows the shapes</text><g fill="#185FA5"><circle cx="201" cy="158" r="3.6"/><circle cx="145" cy="116" r="3.6"/><circle cx="219" cy="129" r="3.6"/><circle cx="160" cy="215" r="3.6"/><circle cx="149" cy="209" r="3.6"/><circle cx="209" cy="139" r="3.6"/><circle cx="186" cy="193" r="3.6"/><circle cx="151" cy="118" r="3.6"/><circle cx="194" cy="181" r="3.6"/><circle cx="174" cy="208" r="3.6"/><circle cx="141" cy="114" r="3.6"/><circle cx="154" cy="139" r="3.6"/><circle cx="165" cy="201" r="3.6"/><circle cx="139" cy="214" r="3.6"/><circle cx="151" cy="132" r="3.6"/><circle cx="140" cy="106" r="3.6"/><circle cx="191" cy="176" r="3.6"/><circle cx="169" cy="209" r="3.6"/><circle cx="153" cy="214" r="3.6"/><circle cx="121" cy="193" r="3.6"/><circle cx="182" cy="201" r="3.6"/><circle cx="135" cy="220" r="3.6"/><circle cx="117" cy="179" r="3.6"/><circle cx="201" cy="168" r="3.6"/><circle cx="158" cy="226" r="3.6"/><circle cx="212" cy="145" r="3.6"/><circle cx="200" cy="164" r="3.6"/><circle cx="196" cy="186" r="3.6"/><circle cx="195" cy="200" r="3.6"/><circle cx="211" cy="138" r="3.6"/><circle cx="148" cy="111" r="3.6"/><circle cx="130" cy="197" r="3.6"/><circle cx="123" cy="202" r="3.6"/><circle cx="214" cy="130" r="3.6"/><circle cx="156" cy="160" r="3.6"/><circle cx="124" cy="205" r="3.6"/><circle cx="140" cy="198" r="3.6"/><circle cx="151" cy="151" r="3.6"/><circle cx="153" cy="220" r="3.6"/><circle cx="206" cy="158" r="3.6"/><circle cx="137" cy="109" r="3.6"/><circle cx="181" cy="203" r="3.6"/><circle cx="147" cy="170" r="3.6"/><circle cx="174" cy="203" r="3.6"/><circle cx="153" cy="147" r="3.6"/></g><g fill="#A32D2D"><circle cx="90" cy="67" r="3.6"/><circle cx="111" cy="189" r="3.6"/><circle cx="42" cy="150" r="3.6"/><circle cx="106" cy="152" r="3.6"/><circle cx="107" cy="178" r="3.6"/><circle cx="137" cy="89" r="3.6"/><circle cx="118" cy="71" r="3.6"/><circle cx="107" cy="182" r="3.6"/><circle cx="47" cy="130" r="3.6"/><circle cx="57" cy="91" r="3.6"/><circle cx="128" cy="90" r="3.6"/><circle cx="98" cy="128" r="3.6"/><circle cx="92" cy="69" r="3.6"/><circle cx="105" cy="165" r="3.6"/><circle cx="101" cy="124" r="3.6"/><circle cx="99" cy="131" r="3.6"/><circle cx="62" cy="89" r="3.6"/><circle cx="60" cy="109" r="3.6"/><circle cx="119" cy="85" r="3.6"/><circle cx="82" cy="78" r="3.6"/><circle cx="48" cy="112" r="3.6"/><circle cx="53" cy="118" r="3.6"/><circle cx="110" cy="66" r="3.6"/><circle cx="74" cy="85" r="3.6"/><circle cx="99" cy="164" r="3.6"/><circle cx="74" cy="77" r="3.6"/><circle cx="46" cy="125" r="3.6"/><circle cx="97" cy="136" r="3.6"/><circle cx="121" cy="87" r="3.6"/><circle cx="57" cy="99" r="3.6"/><circle cx="41" cy="162" r="3.6"/><circle cx="69" cy="85" r="3.6"/><circle cx="100" cy="153" r="3.6"/><circle cx="97" cy="73" r="3.6"/><circle cx="88" cy="75" r="3.6"/><circle cx="125" cy="71" r="3.6"/><circle cx="44" cy="131" r="3.6"/><circle cx="43" cy="160" r="3.6"/><circle cx="113" cy="76" r="3.6"/><circle cx="52" cy="128" r="3.6"/><circle cx="41" cy="141" r="3.6"/><circle cx="115" cy="77" r="3.6"/><circle cx="102" cy="94" r="3.6"/><circle cx="115" cy="184" r="3.6"/><circle cx="89" cy="64" r="3.6"/></g><g fill="#185FA5"><circle cx="461" cy="158" r="3.6"/><circle cx="479" cy="129" r="3.6"/><circle cx="420" cy="215" r="3.6"/><circle cx="371" cy="189" r="3.6"/><circle cx="409" cy="209" r="3.6"/><circle cx="469" cy="139" r="3.6"/><circle cx="446" cy="193" r="3.6"/><circle cx="366" cy="152" r="3.6"/><circle cx="454" cy="181" r="3.6"/><circle cx="367" cy="178" r="3.6"/><circle cx="434" cy="208" r="3.6"/><circle cx="425" cy="201" r="3.6"/><circle cx="399" cy="214" r="3.6"/><circle cx="367" cy="182" r="3.6"/><circle cx="358" cy="128" r="3.6"/><circle cx="365" cy="165" r="3.6"/><circle cx="361" cy="124" r="3.6"/><circle cx="359" cy="131" r="3.6"/><circle cx="451" cy="176" r="3.6"/><circle cx="429" cy="209" r="3.6"/><circle cx="413" cy="214" r="3.6"/><circle cx="381" cy="193" r="3.6"/><circle cx="442" cy="201" r="3.6"/><circle cx="395" cy="220" r="3.6"/><circle cx="377" cy="179" r="3.6"/><circle cx="461" cy="168" r="3.6"/><circle cx="418" cy="226" r="3.6"/><circle cx="472" cy="145" r="3.6"/><circle cx="460" cy="164" r="3.6"/><circle cx="456" cy="186" r="3.6"/><circle cx="359" cy="164" r="3.6"/><circle cx="455" cy="200" r="3.6"/><circle cx="357" cy="136" r="3.6"/><circle cx="471" cy="138" r="3.6"/><circle cx="390" cy="197" r="3.6"/><circle cx="383" cy="202" r="3.6"/><circle cx="474" cy="130" r="3.6"/><circle cx="360" cy="153" r="3.6"/><circle cx="384" cy="205" r="3.6"/><circle cx="400" cy="198" r="3.6"/><circle cx="413" cy="220" r="3.6"/><circle cx="466" cy="158" r="3.6"/><circle cx="441" cy="203" r="3.6"/><circle cx="434" cy="203" r="3.6"/><circle cx="375" cy="184" r="3.6"/></g><g fill="#A32D2D"><circle cx="405" cy="116" r="3.6"/><circle cx="350" cy="67" r="3.6"/><circle cx="302" cy="150" r="3.6"/><circle cx="411" cy="118" r="3.6"/><circle cx="397" cy="89" r="3.6"/><circle cx="401" cy="114" r="3.6"/><circle cx="414" cy="139" r="3.6"/><circle cx="378" cy="71" r="3.6"/><circle cx="307" cy="130" r="3.6"/><circle cx="317" cy="91" r="3.6"/><circle cx="411" cy="132" r="3.6"/><circle cx="388" cy="90" r="3.6"/><circle cx="352" cy="69" r="3.6"/><circle cx="400" cy="106" r="3.6"/><circle cx="322" cy="89" r="3.6"/><circle cx="320" cy="109" r="3.6"/><circle cx="379" cy="85" r="3.6"/><circle cx="342" cy="78" r="3.6"/><circle cx="308" cy="112" r="3.6"/><circle cx="313" cy="118" r="3.6"/><circle cx="370" cy="66" r="3.6"/><circle cx="334" cy="85" r="3.6"/><circle cx="334" cy="77" r="3.6"/><circle cx="306" cy="125" r="3.6"/><circle cx="381" cy="87" r="3.6"/><circle cx="317" cy="99" r="3.6"/><circle cx="301" cy="162" r="3.6"/><circle cx="408" cy="111" r="3.6"/><circle cx="329" cy="85" r="3.6"/><circle cx="416" cy="160" r="3.6"/><circle cx="411" cy="151" r="3.6"/><circle cx="357" cy="73" r="3.6"/><circle cx="348" cy="75" r="3.6"/><circle cx="385" cy="71" r="3.6"/><circle cx="304" cy="131" r="3.6"/><circle cx="303" cy="160" r="3.6"/><circle cx="373" cy="76" r="3.6"/><circle cx="312" cy="128" r="3.6"/><circle cx="397" cy="109" r="3.6"/><circle cx="301" cy="141" r="3.6"/><circle cx="375" cy="77" r="3.6"/><circle cx="362" cy="94" r="3.6"/><circle cx="407" cy="170" r="3.6"/><circle cx="413" cy="147" r="3.6"/><circle cx="349" cy="64" r="3.6"/></g></svg></div>

The price: results depend on `eps` / `min_samples`, and a single `eps` struggles when
different clusters have **very different densities**.

## 8. Soft clustering — Gaussian Mixture Models

k-means gives **hard** assignments — each point is 100% in one cluster. A **Gaussian
Mixture Model (GMM)** is the **soft** version: it models the data as a blend of several
**Gaussian "blobs"**, each with its own centre **and shape and orientation** (a full
covariance, so clusters can be **stretched, tilted ellipses**, not just circles). Every
point gets a **responsibility** — e.g. "70% cluster A, 30% cluster B" — instead of a single
hard label.

It's fit by **Expectation–Maximization (EM)**, the soft cousin of Lloyd's algorithm:

- **E-step** — given the current Gaussians, compute each point's responsibilities.
- **M-step** — given those responsibilities, update each Gaussian's mean, covariance and weight.

Repeat to convergence. In fact **k-means is just the special case** of a GMM with hard
assignments and equal spherical covariances — so a GMM handles elliptical, overlapping
clusters that k-means can't, and attaches a probability to every assignment.

## 9. Dimensionality reduction — why

Real data often has **hundreds or thousands of features**, and that hurts three ways:

- **The curse of dimensionality** — in high dimensions data becomes sparse and *all*
  pairwise distances look almost equal, so "nearest neighbor" and clustering lose meaning.
- **You can't see it** — humans can't picture beyond 3-D, so any visual exploration needs a
  2-D or 3-D view.
- **Cost & noise** — more features mean more storage, slower training, and more chances to
  overfit on redundant or irrelevant columns.

**Dimensionality reduction** finds a smaller set of features — new *combinations* of the
originals — that keeps most of the information. Uses: **visualization**, **compression**,
**denoising**, and **preprocessing** that speeds up and stabilizes the models downstream.

## 10. PCA, made precise

**Principal Component Analysis** is the workhorse linear method. The idea: find the
directions along which the data **varies the most**, and keep just those.

1. **Center** the data (subtract each feature's mean).
2. Form the **covariance matrix** $C = \tfrac1n X^\top X$.
3. Its **eigenvectors** are the **principal components** — orthogonal directions of maximum
   variance — ordered by eigenvalue $\lambda_i$ (the variance along each). Equivalently take
   the **SVD** $X = U\Sigma V^\top$; the columns of $V$ are the components.
4. **Project** the data onto the top few components.

PC1 is the single direction with the most spread; PC2 is the most-spread direction
*perpendicular* to PC1, and so on:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 360 300" style="width:100%;max-width:360px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A tilted 2-D point cloud with its principal components drawn as axes through the centroid: PC1 is the long axis along the direction of greatest spread (95% of the variance), PC2 the short perpendicular axis (5%)."><defs><marker id="pcah" markerWidth="8" markerHeight="8" refX="4" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#33312E"/></marker><marker id="pcah2" markerWidth="8" markerHeight="8" refX="4" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#9A6A2A"/></marker></defs><rect x="1" y="1" width="358" height="298" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g fill="#185FA5"><circle cx="169" cy="157" r="3.4"/><circle cx="187" cy="176" r="3.4"/><circle cx="147" cy="175" r="3.4"/><circle cx="209" cy="136" r="3.4"/><circle cx="172" cy="162" r="3.4"/><circle cx="185" cy="151" r="3.4"/><circle cx="224" cy="138" r="3.4"/><circle cx="206" cy="138" r="3.4"/><circle cx="191" cy="159" r="3.4"/><circle cx="227" cy="137" r="3.4"/><circle cx="192" cy="158" r="3.4"/><circle cx="126" cy="180" r="3.4"/><circle cx="324" cy="101" r="3.4"/><circle cx="202" cy="155" r="3.4"/><circle cx="179" cy="159" r="3.4"/><circle cx="108" cy="220" r="3.4"/><circle cx="190" cy="128" r="3.4"/><circle cx="157" cy="167" r="3.4"/><circle cx="227" cy="156" r="3.4"/><circle cx="182" cy="159" r="3.4"/><circle cx="250" cy="129" r="3.4"/><circle cx="202" cy="162" r="3.4"/><circle cx="194" cy="152" r="3.4"/><circle cx="193" cy="161" r="3.4"/><circle cx="157" cy="163" r="3.4"/><circle cx="184" cy="171" r="3.4"/><circle cx="163" cy="179" r="3.4"/><circle cx="161" cy="188" r="3.4"/><circle cx="151" cy="180" r="3.4"/><circle cx="247" cy="126" r="3.4"/><circle cx="186" cy="155" r="3.4"/><circle cx="243" cy="140" r="3.4"/><circle cx="186" cy="165" r="3.4"/><circle cx="174" cy="155" r="3.4"/><circle cx="260" cy="110" r="3.4"/><circle cx="140" cy="190" r="3.4"/><circle cx="219" cy="128" r="3.4"/><circle cx="172" cy="154" r="3.4"/><circle cx="218" cy="159" r="3.4"/><circle cx="199" cy="157" r="3.4"/><circle cx="155" cy="174" r="3.4"/><circle cx="272" cy="123" r="3.4"/><circle cx="146" cy="173" r="3.4"/><circle cx="218" cy="139" r="3.4"/><circle cx="167" cy="163" r="3.4"/><circle cx="150" cy="176" r="3.4"/><circle cx="197" cy="155" r="3.4"/><circle cx="164" cy="203" r="3.4"/><circle cx="196" cy="152" r="3.4"/><circle cx="250" cy="115" r="3.4"/><circle cx="211" cy="131" r="3.4"/><circle cx="190" cy="146" r="3.4"/><circle cx="134" cy="184" r="3.4"/><circle cx="241" cy="146" r="3.4"/><circle cx="115" cy="202" r="3.4"/><circle cx="219" cy="136" r="3.4"/><circle cx="192" cy="143" r="3.4"/><circle cx="189" cy="156" r="3.4"/><circle cx="218" cy="133" r="3.4"/><circle cx="230" cy="123" r="3.4"/></g><line x1="286" y1="103" x2="99" y2="208" stroke="#33312E" stroke-width="2.6" marker-start="url(#pcah)" marker-end="url(#pcah)"/><line x1="177" y1="138" x2="209" y2="173" stroke="#9A6A2A" stroke-width="2.4" marker-start="url(#pcah2)" marker-end="url(#pcah2)"/><circle cx="193" cy="155" r="3" fill="#33312E"/><text x="96" y="223" font-family="sans-serif" font-size="11.5" fill="#33312E">PC1 (95% var)</text><text x="214" y="182" font-family="sans-serif" font-size="11.5" fill="#9A6A2A">PC2 (5%)</text></svg></div>

The **explained-variance ratio** $\lambda_i/\sum_j\lambda_j$ says how much each component
keeps — here PC1 alone holds 95%, so projecting onto it loses almost nothing. Pick enough
components to reach, say, **95% cumulative** variance (read it off a *scree plot*).
Standardize first so large-scale features don't hijack the variance. *(This is the Math
module's eigenvectors / SVD / projection in action.)*

## 11. Non-linear embeddings — t-SNE & UMAP

PCA is **linear**, so it can flatten curved structure together. **t-SNE** and **UMAP** are
non-linear methods built for **visualization**: they map high-dim data down to 2-D while
trying to keep **neighbors close** — points near each other stay near — which reveals
clusters that PCA might hide. These produce the striking 2-D "maps" you see of image and
word datasets.

Read them with care:
- the **size** of a cluster and the **distance between** clusters are **not meaningful**;
- results **shift** with the hyperparameters (perplexity / `n_neighbors`) and the seed;
- they're for **looking**, not as features fed into another model — use PCA for that.

## 12. Anomaly detection

Flag points that don't fit the learned structure: far from any centroid, in a
low-density region (DBSCAN noise), low GMM likelihood, or via dedicated methods
(Isolation Forest, one-class SVM). Used heavily for fraud, fault, and intrusion detection.

## 13. Evaluating without labels

With no ground truth there's no accuracy, so judge the structure three ways:

- **Internal indices** — score the geometry itself: **inertia**, the **silhouette** (§4),
  Davies–Bouldin. Cheap, but they tend to reward k-means-shaped clusters.
- **Stability** — re-run on bootstraps / subsamples / different seeds; structure that's
  *real* reappears, structure that's noise doesn't.
- **The downstream task** (most honest) — do the segments improve targeting? do the PCA
  features help the classifier? **Usefulness**, not a single number, is the real test.

## 14. Where it connects

Customer segmentation, anomaly/fraud detection, compression, and — crucially —
**representation learning**: the **embeddings** behind search, recommendation, and LLMs
are learned low-dimensional vectors. Unsupervised and **self-supervised** structure (M0
§2) is how modern AI learns from oceans of unlabeled data.
"""

_M5_TASKS = r"""
### Warm-up — in the Playground tab
1. Generate 3 true blobs but set **k = 2**, then **k = 4** — how do the clusters and the
   **inertia** change? Which k looks right?
2. Increase the **overlap** between blobs — when does k-means start making mistakes?
3. Re-run with different **seeds** — does the result change? (initialization sensitivity)

### Pencil & paper
4. Run one k-means iteration by hand on 4 points and 2 centroids.
5. Explain why inertia *always* decreases as k increases (so you can't just minimize it).
6. Compute a point's silhouette given a = 1.0 (own cluster) and b = 3.0 (nearest other).

### Code
7. Implement k-means (assign/update loop) in NumPy and plot the **elbow** curve.
8. Implement PCA via `np.linalg.svd`; project a dataset to 2-D and plot explained variance.
9. Try `sklearn` `KMeans`, `DBSCAN`, and `PCA` on the same data and compare.

### Bridge
10. PCA's components are eigenvectors of the covariance matrix — connect this to the
    Math module (vectors, dot products, projection, SVD).
"""

_M5_REFS = r"""
### Books
- James, Witten, Hastie & Tibshirani — *ISL*, ch. 12 (unsupervised) — [free PDF](https://www.statlearning.com/).
- Hastie, Tibshirani & Friedman — *ESL*, ch. 14. · Bishop — *PRML*, ch. 9 (k-means, GMM/EM).

### Docs & reading
- scikit-learn — [clustering](https://scikit-learn.org/stable/modules/clustering.html) & [decomposition / PCA](https://scikit-learn.org/stable/modules/decomposition.html).
- van der Maaten & Hinton (2008) — *t-SNE*. · McInnes et al. (2018) — *UMAP*.

### In this lab
- Math module: vectors, projection, SVD underpin PCA. ANN module: embeddings.
"""

_M5_QUIZ = [
    Question("Unsupervised learning works with…",
             ["labeled examples (x, y)", "inputs only (no labels), to find structure",
              "rewards", "a single data point"], 1,
             "No labels — clustering, dimensionality reduction, anomaly detection."),
    Question("k-means alternates which two steps?",
             ["forward and backward passes", "assign points to nearest centroid, then move centroids to the mean",
              "split and prune", "encode and decode"], 1,
             "Lloyd's algorithm: assignment step then update step, minimizing inertia."),
    Question("A limitation of k-means is that it…",
             ["needs labels", "requires you to choose k and assumes roughly spherical clusters",
              "can't use numeric data", "always finds the global optimum"], 1,
             "k is fixed in advance, clusters are assumed spherical/equal-size, and it's init-sensitive."),
    Question("PCA finds directions that…",
             ["maximize classification accuracy", "maximize the variance of the data (principal components)",
              "minimize the number of points", "are always axis-aligned"], 1,
             "PCA's components are the orthogonal max-variance directions (top eigenvectors / SVD)."),
    Question("The 'elbow' and 'silhouette' methods are used to…",
             ["train faster", "choose the number of clusters k", "scale features", "label the data"], 1,
             "They help pick k since there's no ground-truth label to validate against."),
    Question("t-SNE / UMAP are mainly for…",
             ["supervised prediction", "visualizing high-dimensional data in 2-D (preserving local neighborhoods)",
              "feature scaling", "boosting"], 1,
             "They're non-linear embeddings for visualization; cluster sizes/distances aren't meaningful."),
    Question("Why can't you evaluate clustering with 'accuracy'?",
             ["clustering is always perfect", "there are no ground-truth labels to compare against",
              "accuracy needs probabilities", "it's too slow"], 1,
             "Unsupervised = no labels; use internal metrics (inertia/silhouette) or downstream usefulness."),
    Question("k-means++ and multiple restarts are used because k-means…",
             ["needs labels", "only converges to a local minimum, so initialization matters",
              "can't handle 2-D data", "has no objective"], 1,
             "Lloyd's algorithm finds a local optimum; better/spread-out seeds and restarts improve the result."),
]

UNSUPERVISED = Lesson("unsupervised", "Unsupervised learning", _M5_THEORY, _M5_QUIZ, _M5_TASKS, _M5_REFS)


# ===========================================================================
# M6 — Model selection & validation
# ===========================================================================

_M6_THEORY = r"""
## 1. The problem

You have many candidate models and hyperparameters. **How do you choose without
cheating?** Whatever scores best on the *test* set will look good there *by selection*,
so reusing the test set for choices makes your final number a lie. Model selection is
the discipline of choosing fairly — and getting an **honest** estimate of future
performance.

## 2. Validation & cross-validation

- **Hold-out validation** — one train/validation/test split. Simple, but the estimate is
  **noisy**: it depends on which rows happened to land in validation.
- **k-fold cross-validation** — split train into $k$ folds; train on $k-1$, validate on
  the held-out fold, **rotate $k$ times, average**. Far more stable and uses all data.
  Use **stratified** folds for classification (preserve class ratios). $k=5$ or $10$ is
  standard.

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 480 250" style="width:100%;max-width:480px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="5-fold cross-validation: the data is split into 5 equal parts; across 5 rounds each part is used once as the validation set while the other four train the model, and the scores are averaged."><rect x="1" y="1" width="478" height="248" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><text x="16" y="22" font-family="sans-serif" font-size="13" fill="#33312E">5-fold cross-validation</text><rect x="300" y="11" width="12" height="12" rx="2" fill="#EBB661"/><text x="316" y="21" font-family="sans-serif" font-size="11" fill="#6B6A66">validation</text><rect x="398" y="11" width="12" height="12" rx="2" fill="#E3E6EA"/><text x="414" y="21" font-family="sans-serif" font-size="11" fill="#6B6A66">train</text><g stroke="#FFFFFF"><rect x="110" y="30" width="68" height="30" rx="3" fill="#EBB661"/><rect x="182" y="30" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="254" y="30" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="326" y="30" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="398" y="30" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="110" y="68" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="182" y="68" width="68" height="30" rx="3" fill="#EBB661"/><rect x="254" y="68" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="326" y="68" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="398" y="68" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="110" y="106" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="182" y="106" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="254" y="106" width="68" height="30" rx="3" fill="#EBB661"/><rect x="326" y="106" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="398" y="106" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="110" y="144" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="182" y="144" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="254" y="144" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="326" y="144" width="68" height="30" rx="3" fill="#EBB661"/><rect x="398" y="144" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="110" y="182" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="182" y="182" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="254" y="182" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="326" y="182" width="68" height="30" rx="3" fill="#E3E6EA"/><rect x="398" y="182" width="68" height="30" rx="3" fill="#EBB661"/></g><g font-family="sans-serif" font-size="10" fill="#7A4E12" text-anchor="middle"><text x="144" y="50">val</text><text x="216" y="88">val</text><text x="288" y="126">val</text><text x="360" y="164">val</text><text x="432" y="202">val</text></g><g font-family="sans-serif" font-size="11" fill="#6B6A66" text-anchor="end"><text x="102" y="50">round 1</text><text x="102" y="88">round 2</text><text x="102" y="126">round 3</text><text x="102" y="164">round 4</text><text x="102" y="202">round 5</text></g><text x="245" y="236" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#9C9B95">Each part is the validation set once — average the 5 scores for a stable estimate.</text></svg></div>
- **Leave-one-out (LOOCV)** — $k=n$; nearly unbiased but **high variance** and expensive.
- **Repeated** k-fold averages several shuffles for an even steadier estimate.

There's a bias–variance tradeoff in $k$ itself: small $k$ → each train set is smaller →
**pessimistic bias**; large $k$ → estimates **highly correlated** + costly. The **test
set is never touched** during any of this.

## 3. A worked example

5-fold CV on **1000** rows: each round trains on **800** and validates on **200**;
do it **5 times** (each row is in validation exactly once), then average the 5 scores.
If a hyperparameter grid has 12 settings, that's $12\times5 = 60$ model fits to pick the
best setting — then you refit on all 1000 and report once on the sealed test set.

## 4. Hyperparameter search

Hyperparameters (learning rate, tree depth, C, γ, k, λ) are **chosen on validation, not
learned**. Strategies:
- **Grid search** — try every combination on a grid. Simple, but cost explodes
  combinatorially (the curse of dimensionality, again).
- **Random search** — sample combinations at random. Usually **more efficient**: when
  only a few hyperparameters matter, random sampling covers those important axes with
  far fewer trials than a dense grid.
- **Bayesian optimization / Hyperband / successive halving** — model the search and spend
  trials where they're most promising; best when each fit is expensive.

## 5. Nested cross-validation

If you tune **and** report on the *same* CV, the reported score is optimistic (you chose
the luckiest setting). **Nested CV** uses an **inner** loop to tune and an **outer** loop
to evaluate — giving an honest estimate when data is too scarce for a separate test set.
Costlier, but the gold standard for small-data model comparison.

## 6. Regularization — continuous model selection

Rather than picking among discrete models, **regularization** dials complexity smoothly:
**L2 (ridge)** shrinks weights, **L1 (lasso)** zeroes some out (feature selection),
**elastic net** blends both, **early stopping** halts before overfitting, **dropout**
regularizes nets. The strength $\lambda$ is itself a hyperparameter chosen on validation.

## 7. Diagnosing with curves

- **Learning curve** — error vs. **training-set size**. Train & validation curves
  converging to a **high** error ⇒ **high bias** (more data won't help — get a richer
  model). A large **gap** that shrinks as data grows ⇒ **high variance** (more data helps).
- **Validation curve** — error vs. a single **hyperparameter** (e.g. tree depth, C). It
  shows the under-fit → sweet-spot → over-fit arc directly, so you can read off the best
  value.

## 8. Pipelines — fit inside the fold

The most common subtle leak: scaling, imputing, or feature-selecting on the **whole**
dataset before CV — the validation rows then influence preprocessing. Fix: wrap
preprocessing + model in a **Pipeline** so every transformer is **fit only on each fold's
training part**. Then CV honestly simulates "new data arriving."

## 9. Picking the final model

Use the validation metric that matches your real goal (M0 §8). The **one-standard-error
rule**: among models within 1 standard error of the best CV score, prefer the
**simplest** one — it's usually more robust and less likely to be a lucky fluke. Consider
**calibration** (M2 §12) if you act on the probabilities.

## 10. Pitfalls

- **Data leakage** via preprocessing or target-derived features — the big one.
- **Overfitting the validation set** by trying too many things (a form of p-hacking /
  multiple comparisons): the more configurations you test, the more one wins by luck.
- **Distribution shift** between your splits and production data.
- **Inconsistent folds** — always compare candidate models on the *same* CV splits.

## 11. A practical recipe

Split off a test set → build a Pipeline → choose a model family → tune hyperparameters
with (stratified) k-fold CV (random search first, refine with grid) → diagnose with
learning/validation curves → refit the best config on all training data → report **once**
on the test set vs. a baseline → keep the simplest model within 1 SE of the best.
"""

_M6_TASKS = r"""
### Pencil & paper
1. With 1000 rows and 5-fold CV, how many models are trained, and how big is each
   train/validation part?
2. Sketch learning curves for (a) a high-bias model, (b) a high-variance model.
3. Explain the one-standard-error rule and why it favors simpler models.

### Code
4. Implement k-fold CV by hand (NumPy) and average the scores; compare to a single split's noise.
5. Use `sklearn` `Pipeline(StandardScaler, LogisticRegression)` inside `cross_val_score`
   and show it differs from scaling-then-CV (leakage).
6. Run `GridSearchCV` vs `RandomizedSearchCV` over C/γ for an SVM; compare cost & result.
7. Plot a **validation curve** (score vs. tree depth) and a **learning curve** (score vs. data size).

### Concept
8. Explain why touching the test set during tuning invalidates the final estimate.
"""

_M6_REFS = r"""
### Books
- James, Witten, Hastie & Tibshirani — *ISL*, ch. 5 (resampling: CV & bootstrap) — [free PDF](https://www.statlearning.com/).
- Hastie, Tibshirani & Friedman — *ESL*, ch. 7 (model assessment & selection).

### Docs
- scikit-learn — [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [tuning](https://scikit-learn.org/stable/modules/grid_search.html), [pipelines](https://scikit-learn.org/stable/modules/compose.html).
- Bergstra & Bengio (2012) — *Random Search for Hyper-Parameter Optimization*.

### In this lab
- ML: M0 (split, bias–variance, leakage), and the M0 overfitting playground.
"""

_M6_QUIZ = [
    Question("The main reason for a validation set (separate from test) is to…",
             ["train the parameters", "choose models/hyperparameters without touching the test set",
              "increase the dataset", "compute accuracy"], 1,
             "Validation is for selection; the test set stays sealed for the final estimate."),
    Question("k-fold cross-validation gives a better estimate than a single split because it…",
             ["uses the test set", "averages over k rotations, reducing the noise of one split",
              "trains only once", "needs no data"], 1,
             "Rotating the held-out fold and averaging is far more stable than one hold-out."),
    Question("Random search often beats grid search because…",
             ["it tries more combinations", "it spends trials more efficiently when only a few hyperparameters matter",
              "it never overfits", "it's exhaustive"], 1,
             "Random search covers important dimensions better per trial than a dense grid."),
    Question("Scaling the whole dataset before doing CV causes…",
             ["faster training", "data leakage — the validation folds influence preprocessing",
              "underfitting", "no problem at all"], 1,
             "Fit transformers inside each fold (use a Pipeline) so CV honestly simulates new data."),
    Question("Learning curves that converge to a HIGH error as data grows indicate…",
             ["high variance (more data helps)", "high bias (a better/more-complex model is needed)",
              "perfect fit", "leakage"], 1,
             "High-bias models plateau high; more data won't fix underfitting."),
    Question("Repeatedly tuning against the same validation set risks…",
             ["nothing", "overfitting the validation set (optimistic estimates)",
              "underfitting", "slower inference"], 1,
             "Trying many things on one validation set is a form of p-hacking; use nested CV / a final test."),
    Question("Nested cross-validation is used to…",
             ["speed up training", "get an honest performance estimate while also tuning, when data is scarce",
              "avoid scaling", "replace the model"], 1,
             "Inner loop tunes, outer loop evaluates — so the reported score isn't optimistic from selection."),
]

MODEL_SELECTION = Lesson("model_selection", "Model selection & validation",
                         _M6_THEORY, _M6_QUIZ, _M6_TASKS, _M6_REFS)


# ===========================================================================
# M7 — Practical ML
# ===========================================================================

_M7_THEORY = r"""
## 1. The reality of ML work

Most of a real project is **not** modeling — it's getting, cleaning, and shaping data,
then plumbing it into production (M0 §1). A simple model on great features beats a fancy
model on bad ones, every time. This module is the unglamorous craft that decides whether
a project actually works.

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 720 160" style="width:100%;max-width:720px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The end-to-end machine learning workflow: Data, Clean and split, Features, Train, Validate and tune, Deploy and monitor, connected left to right, with a dashed feedback loop from monitoring back to features."><defs><marker id="pwah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#5B8FC2"/></marker><marker id="pwah2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#9A6A2A"/></marker></defs><rect x="1" y="1" width="718" height="158" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><path d="M655,55 C655,20 301,20 301,55" fill="none" stroke="#9A6A2A" stroke-width="1.6" stroke-dasharray="5 3" marker-end="url(#pwah2)"/><text x="478" y="15" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#9A6A2A">monitoring → new data → retrain</text><g stroke="#5B8FC2" stroke-width="2"><line x1="115" y1="80" x2="131" y2="80" marker-end="url(#pwah)"/><line x1="233" y1="80" x2="249" y2="80" marker-end="url(#pwah)"/><line x1="351" y1="80" x2="367" y2="80" marker-end="url(#pwah)"/><line x1="469" y1="80" x2="485" y2="80" marker-end="url(#pwah)"/><line x1="587" y1="80" x2="603" y2="80" marker-end="url(#pwah)"/></g><g><rect x="15" y="55" width="100" height="50" rx="8" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.6"/><rect x="133" y="55" width="100" height="50" rx="8" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.6"/><rect x="251" y="55" width="100" height="50" rx="8" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.6"/><rect x="369" y="55" width="100" height="50" rx="8" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.8"/><rect x="487" y="55" width="100" height="50" rx="8" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.6"/><rect x="605" y="55" width="100" height="50" rx="8" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.6"/></g><g font-family="sans-serif" font-size="12.5" text-anchor="middle" fill="#0C447C"><text x="65" y="84">Data</text><text x="183" y="78">Clean &amp;</text><text x="183" y="94">split</text><text x="301" y="84">Features</text><text x="537" y="78">Validate</text><text x="537" y="94">&amp; tune</text><text x="655" y="78">Deploy &amp;</text><text x="655" y="94">monitor</text></g><text x="419" y="84" text-anchor="middle" font-family="sans-serif" font-size="12.5" fill="#5A3E14">Train</text><text x="360" y="145" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#6B6A66">Most of the effort is getting &amp; shaping data and shipping it — “Train” is the small box.</text></svg></div>

## 2. Feature engineering

Turning raw data into informative inputs — often the **highest-leverage** activity:
- **Domain features** — ratios, differences, aggregates an expert would compute
  (e.g. debt-to-income, clicks-per-session).
- **Interactions & non-linear terms** — products of features, polynomials, so linear
  models can capture combined effects.
- **Datetime** — day-of-week, month, is-holiday, time-since-last-event (cyclical encode
  with sin/cos for hour/month).
- **Text** — bag-of-words or **TF-IDF** (term frequency × inverse document frequency:
  $\text{tfidf} = \text{tf}\cdot\log\frac{N}{df}$, which down-weights common words), or
  learned embeddings.
- **Binning** — bucket a continuous feature when the relationship is non-linear or you
  want robustness to outliers.

## 3. Scaling & normalization

- **Standardization** $\frac{x-\mu}{\sigma}$ (mean 0, std 1), **min-max** to $[0,1]$, or
  **robust** scaling (median/IQR) when outliers are present.
- **Needed** for distance- and gradient-based models: **SVM, k-NN, k-means, PCA, neural
  nets, regularized linear models.**
- **Not needed** for **trees / forests / boosting** (split rules are scale-invariant).
- Always **fit the scaler on train only** and apply to val/test (M6 pipelines).

## 4. Missing data

- **Drop** rows/columns — only when little is missing.
- **Impute** — mean/median (numeric), most-frequent (categorical), or model-based
  (kNN / iterative imputation).
- **Missingness as signal** — add an "is-missing" indicator column; that a value is
  absent is often itself predictive (e.g. an unfilled optional field).
- Mechanisms matter (MCAR / MAR / MNAR): if data is missing *because* of its value,
  naive imputation biases the model.

## 5. Categorical encoding

- **One-hot** — a 0/1 column per category; ideal for **low cardinality**.
- **Ordinal** — integers for genuinely ordered categories (small < medium < large).
- **Target / frequency encoding** — replace a high-cardinality category (zip code, user
  id) with a statistic of the target; **powerful but a leakage trap** — compute it
  **inside CV folds** / with smoothing, never on the full data.
- **Hashing** or **learned embeddings** for very high cardinality.

## 6. Imbalanced classes

When positives are rare (fraud, disease) accuracy lies (M0 §8). Fixes:
- **Resample** — oversample the minority (**SMOTE** synthesizes new minority points by
  interpolating between neighbors) or undersample the majority.
- **Class weights** — make minority errors cost more in the loss.
- **Threshold tuning** + the right metric (precision/recall/F1/PR-AUC, M2).
Tune resampling **inside** CV folds — resampling before splitting leaks.

## 7. Outliers & robustness

Decide per feature: clip/winsorize, transform (log for skewed/positive data), use robust
scalers/losses (MAE, Huber), or keep them if they're real signal. Tree models tolerate
outliers in $x$; linear/SVM/k-means do not.

## 8. Data leakage — the cardinal sin (again)

The most common way to ship a model that looked great offline and fails live:
- **preprocessing fit on all data** (scale/encode/impute before splitting);
- features that encode the **target** or use the **future** (a `payment_date` predicting
  `will_pay`; an aggregate computed over the whole dataset including the test period);
- **duplicate or grouped rows** spanning train and test (same patient in both).
Defenses: **split first**, do everything in a **Pipeline**, respect **group/time**
structure (GroupKFold, TimeSeriesSplit), and ask of every feature: *"would this value
exist, unchanged, at prediction time?"*

## 9. The ecosystem

- **pandas / NumPy** — data wrangling.
- **scikit-learn** — `Pipeline` and `ColumnTransformer` (different transforms per column
  type), transformers (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`), estimators,
  `cross_val_score`, `GridSearchCV` — all behind one consistent `fit` / `predict` /
  `transform` API.
- **Reproducibility** — fix random **seeds**; version data, code, and models; log
  experiments (config + metrics) so results are repeatable.

## 10. From notebook to production

A model isn't done when the notebook runs. **Serialize the *whole* pipeline**
(preprocessing + model together), serve it (batch scoring or a real-time API),
**monitor** inputs and predictions for **drift**, and **retrain** on fresh data on a
schedule. The loop (M0 §3) never ends — and operations, not accuracy, is where most
deployed models quietly fail.

## 11. The end-to-end checklist

Frame → look at the data (EDA) → split off a test set → build a Pipeline (impute →
encode → scale → model) → cross-validate & tune → diagnose curves → evaluate **once** on
the sealed test set vs. a baseline → ship the **pipeline** (not just the model) →
monitor & retrain. Keep it simple first; add complexity only when it earns its keep.
"""

_M7_TASKS = r"""
### Hands-on
1. Take a messy CSV (missing values, categoricals, different scales) and build a
   `sklearn` **Pipeline**: impute → one-hot → scale → logistic regression. Cross-validate it.
2. Engineer 3 new features for a dataset you know; measure whether they help (CV).
3. Create an **imbalanced** dataset; compare accuracy vs. F1/PR-AUC, then apply class
   weights or SMOTE (inside the CV folds) and re-measure.

### Spot-the-leak
4. List three concrete ways data leakage could sneak into a churn model, and how a
   Pipeline + "available at prediction time?" test prevents each.

### Concept
5. Which of these need feature scaling and which don't: SVM, random forest, k-NN,
   gradient boosting, k-means, logistic regression? Why?
6. Why must target encoding be computed inside CV folds rather than on the full dataset?

### Bridge
7. Connect: standardization (here) ↔ the Math module (norms), and ↔ why ANN training
   needs normalized inputs / good initialization.
"""

_M7_REFS = r"""
### Books & courses
- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (the practical bible).
- Google — [ML Crash Course](https://developers.google.com/machine-learning/crash-course) & [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml).

### Docs
- scikit-learn — [preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html), [pipelines & ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html), [imputation](https://scikit-learn.org/stable/modules/impute.html).

### In this lab
- ML: M0 (workflow, leakage), M6 (pipelines inside CV). Math: scaling & norms.
"""

_M7_QUIZ = [
    Question("In a typical ML project, the largest share of effort goes to…",
             ["hyperparameter tuning", "data collection, cleaning, and feature engineering",
              "choosing the loss", "writing the API"], 1,
             "Data work dominates; a simple model on good features beats a fancy model on bad data."),
    Question("Feature scaling (standardization) is important for which model?",
             ["random forest", "SVM / k-NN / k-means / neural nets (distance- or gradient-based)",
              "decision tree", "gradient boosting"], 1,
             "Distance- and gradient-based models need scaling; tree-based models are scale-invariant."),
    Question("One-hot encoding is used to…",
             ["scale numeric features", "represent categorical features as binary columns",
              "fill missing values", "reduce dimensionality"], 1,
             "It turns each category into its own 0/1 column so models can use it."),
    Question("On an imbalanced dataset you should…",
             ["rely on accuracy", "use class weights / resampling and metrics like F1 or PR-AUC",
              "drop the minority class", "ignore the threshold"], 1,
             "Accuracy misleads; rebalance and use precision/recall/F1/AUC and threshold tuning."),
    Question("To prevent data leakage you should…",
             ["scale on the full dataset before splitting", "split first and fit all preprocessing inside the pipeline/CV folds",
              "use more features", "train longer"], 1,
             "Fit transformers only on training data (Pipelines) and ask if a feature exists at prediction time."),
    Question("'Missingness as a signal' means…",
             ["always drop missing rows", "the fact that a value is missing can itself be predictive (add an is-missing flag)",
              "impute with zero always", "missing data is harmless"], 1,
             "Sometimes whether a value is absent carries information worth encoding."),
    Question("Target encoding a high-cardinality column must be done inside CV folds because…",
             ["it is faster", "computing it on all data leaks the target into the features",
              "it needs scaling", "one-hot is always better"], 1,
             "Using the target over the whole dataset (incl. validation rows) is leakage; encode within folds / with smoothing."),
]

PRACTICAL = Lesson("practical", "Practical ML", _M7_THEORY, _M7_QUIZ, _M7_TASKS, _M7_REFS)


# ===========================================================================
# M8 — ML in Python (the practical toolkit, with runnable examples)
# ===========================================================================
_M8_THEORY = r"""
## 1. The Python ML stack

Almost all classical ML in Python rests on a handful of libraries:

- **NumPy** — n-dimensional arrays + fast vectorized math. The substrate everything builds on (it's what our `core/` uses).
- **pandas** — labeled tables (`DataFrame`) for loading, cleaning and joining real data (CSV, Excel, SQL, Parquet).
- **scikit-learn** — the workhorse: dozens of models + preprocessing + model-selection tools behind **one consistent API**. For tabular data, this *is* "doing ML in Python".
- **matplotlib** / seaborn — plotting.
- **statsmodels** — classical statistics & inference (p-values, confidence intervals).
- **XGBoost / LightGBM** — gradient-boosted trees; usually the top accuracy on tabular data.
- **PyTorch / TensorFlow / JAX** — deep learning for images, text, audio (the ANN module's world).

**Rule of thumb:** tabular data → scikit-learn (+ XGBoost); images / text / sequences → PyTorch.

## 2. The scikit-learn API — learn it once, use it everywhere

Every model (an *estimator*) obeys the same tiny contract:

- `model.fit(X, y)` — **learn** from data.
- `model.predict(X)` — **predict** labels/values (`predict_proba` gives class probabilities).
- `model.score(X, y)` — a **default metric** (accuracy for classifiers, R² for regressors).

Preprocessors (*transformers*) use `fit` then `transform` (or `fit_transform`). Here `X` is a 2-D
array/DataFrame of shape `(n_samples, n_features)` and `y` is 1-D `(n_samples,)`. Because the API is
uniform, swapping one model for another is a **one-line change**:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)            # learn
preds = model.predict(X_test)          # use
acc   = model.score(X_test, y_test)    # evaluate
```

Change `LogisticRegression()` to `RandomForestClassifier()` or `SVC()` and everything else stays the same.

## 3. The standard workflow

1. **Load** — `df = pandas.read_csv("data.csv")`.
2. **Explore** — `df.describe()`, `df.info()`, plots, check missing values & class balance.
3. **Split** — `train_test_split(...)`; hold out a test set you **never** tune on.
4. **Preprocess** — scale numbers, encode categories, impute missing — **inside a Pipeline**.
5. **Train** — `fit` on the training set.
6. **Evaluate** — accuracy / precision-recall / R² on held-out data; cross-validate.
7. **Tune** — `GridSearchCV` / `RandomizedSearchCV` over hyperparameters.
8. **Persist** — `joblib.dump(model, "model.joblib")` for production.

## 4. A complete classification example

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X, y = load_wine(return_X_y=True)                 # 178 samples, 13 features, 3 classes
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)

model = make_pipeline(StandardScaler(),           # scale, then classify
                      LogisticRegression(max_iter=1000))
model.fit(X_tr, y_tr)

preds = model.predict(X_te)
print("test accuracy:", accuracy_score(y_te, preds))
print(classification_report(y_te, preds))
print(confusion_matrix(y_te, preds))
print("5-fold CV:", cross_val_score(model, X, y, cv=5).mean())
```

The **🧪 Live examples** tab runs exactly this — pick the dataset and model and watch it retrain.

## 5. Pipelines — preprocessing + model as ONE object

Always wrap preprocessing and the model in a `Pipeline`. Two reasons:

- **No data leakage** — the scaler is fit on the *training fold only*, automatically, even inside cross-validation.
- **One object** to fit, predict, save and deploy — production gets the identical transforms.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)
```

For mixed column types, `ColumnTransformer` scales numeric columns and one-hot-encodes categorical ones in one shot.

## 6. Cross-validation & hyperparameter search

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

cross_val_score(pipe, X, y, cv=5)                 # 5 honest held-out scores

grid = GridSearchCV(pipe,
                    {"logisticregression__C": [0.1, 1, 10]},
                    cv=5)
grid.fit(X, y)
print(grid.best_params_, grid.best_score_)
```

That's the M6 cross-validation picture in a few lines of code.

## 7. Evaluating properly

```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
print(classification_report(y_test, preds))       # precision / recall / F1 per class
```

Pick the metric that matches the **cost of errors** (M6): accuracy is misleading on imbalanced data —
prefer precision/recall, F1, or ROC-AUC. For regression use RMSE and R².

## 8. Saving & shipping a model

```python
import joblib
joblib.dump(model, "model.joblib")    # save the fitted PIPELINE
model = joblib.load("model.joblib")   # load later in production
```

Persist the **whole pipeline** (preprocessing + model) so the serving code applies identical transforms.

## 9. When to leave scikit-learn

scikit-learn is ideal for **tabular** data up to ~millions of rows. Reach beyond it for:

- **XGBoost / LightGBM** — usually the best tabular accuracy (boosted trees).
- **PyTorch** — images, text, audio, anything that needs deep nets / GPUs → the **ANN module**.

## 10. The honest 80/20

In real projects ~**80% of effort is data** (M7) — getting, cleaning, joining, building features — and
the modeling above is the easy 20%. Good data with a simple scikit-learn model beats a fancy model on
bad data, every time.
"""

_M8_QUIZ = [
    Question("What are scikit-learn's three core estimator methods?",
             ["`load` / `run` / `report`", "`fit` / `predict` / `score`",
              "`train` / `test` / `deploy`", "`compile` / `fit` / `evaluate`"], 1,
             "`fit` learns, `predict` uses the model, `score` gives a default metric. Transformers add `transform`."),
    Question("Why wrap preprocessing and the model in a Pipeline?",
             ["It runs faster", "It prevents data leakage (the scaler is fit on the training fold only) and bundles everything into one object",
              "It is required by Python", "It removes the need for a test set"], 1,
             "A Pipeline fits preprocessing on training data only — automatically correct inside cross-validation — and saves/deploys as one object."),
    Question("For tabular data the go-to library is ___; for images/text it's ___.",
             ["PyTorch; scikit-learn", "pandas; NumPy",
              "scikit-learn; PyTorch", "statsmodels; XGBoost"], 2,
             "scikit-learn dominates classical/tabular ML; deep learning on images/text/audio uses PyTorch (or TF/JAX)."),
    Question("What does `train_test_split` give you?",
             ["A faster model", "A held-out test set you never tune on, for an honest performance estimate",
              "Automatic feature selection", "Cross-validation folds"], 1,
             "It carves out data the model never sees during training/tuning, so the reported score isn't optimistic."),
    Question("Roughly how is effort split in a real ML project?",
             ["80% modeling, 20% data", "50/50",
              "~80% data wrangling & features, ~20% modeling", "100% modeling"], 2,
             "Most of the work is getting and shaping data (M7). A simple model on great data usually wins."),
]

_M8_TASKS = r"""
- In the **Live examples** tab, switch the dataset and the model and watch accuracy + the confusion
  matrix change — note which models need scaling (SVM, logistic) and which don't (random forest).
- Copy the §4 example into a notebook (or the 🐍 Sandbox if enabled) and **swap** `LogisticRegression`
  for `RandomForestClassifier` — change one line, rerun.
- Add a `GridSearchCV` over `C` (logistic) or `n_estimators`/`max_depth` (forest); print `best_params_`.
- Load a **CSV of your own** with `pandas.read_csv`, then `df.info()` / `df.describe()` before modeling.
- Wrap a `ColumnTransformer` (scale numeric + one-hot categorical) inside a Pipeline on a mixed-type dataset.
- `joblib.dump` your fitted pipeline, reload it, and confirm `predict` gives identical results.
"""

_M8_REFS = r"""
- **scikit-learn User Guide** — scikit-learn.org/stable/user_guide.html (the single best reference).
- **scikit-learn "Getting Started"** + the *Choosing the right estimator* cheat-sheet.
- Géron, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3rd ed.) — the standard practical book.
- VanderPlas, *Python Data Science Handbook* (free online) — NumPy / pandas / matplotlib / sklearn.
- **pandas** docs — *10 minutes to pandas*; **XGBoost** docs for boosted trees.
- Ties to this lab: M0–M7 (the concepts), and the **ANN module** for the PyTorch/deep-learning path.
"""

PYTHON_ML = Lesson("python_ml", "ML in Python", _M8_THEORY, _M8_QUIZ, _M8_TASKS, _M8_REFS)
