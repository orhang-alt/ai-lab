"""ML module lesson content. Reuses the Lesson/Question/quiz engine from lessons.py."""

from __future__ import annotations

from lessons import Lesson, Question

_THEORY = r"""
## 1. What linear regression does

**The goal:** predict a **continuous** number $y$ from one or more input features. The
model is the simplest possible — a straight line (1 feature) or a flat **hyperplane**
(many features):

$$ \hat y = w_1 x_1 + \dots + w_n x_n + b = \mathbf w\cdot\mathbf x + b. $$

With a single feature it's the line from school, $\hat y = wx + b$:
- $w$ — the **slope**: how much $\hat y$ changes per one-unit increase in $x$.
- $b$ — the **intercept**: the prediction when $x = 0$.

"Fitting" = finding the $w, b$ whose line passes **as close as possible** to the data
(closeness defined in §2). Example: with $\hat y = 2x + 1$, an input $x = 3$ predicts
$\hat y = 7$, and the slope of 2 means "+1 in $x \Rightarrow +2$ in $\hat y$".

This is the *same functional form* as a neuron with a **linear** activation — the
bridge between this ML module and the ANN module.

## 2. The loss — measuring "close"

For each point the **residual** is the vertical gap $\hat y_i - y_i$ (prediction minus
truth). We summarize all residuals into one number, the **mean squared error**:

$$ \text{MSE} = \frac1m\sum_{i=1}^{m}(\hat y_i - y_i)^2 . $$

Picture it: the line is the model, and each **pink segment is one residual** — the
vertical gap to a data point. MSE is the **average of the squares** of those gaps, so
fitting = choosing the line that makes the pink segments as short as possible overall.

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 480 320" style="width:100%;max-width:470px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Scatter of eight data points with the best-fit straight line; each pink vertical segment is a residual, the gap between a point and the line."><rect x="1" y="1" width="478" height="318" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="60" y1="270" x2="445" y2="270" stroke="#C9C8C1" stroke-width="1.2"/><line x1="60" y1="270" x2="60" y2="45" stroke="#C9C8C1" stroke-width="1.2"/><g stroke="#C0507A" stroke-width="1.8"><line x1="102" y1="194" x2="102" y2="208"/><line x1="144" y1="216" x2="144" y2="193"/><line x1="187" y1="158" x2="187" y2="177"/><line x1="229" y1="187" x2="229" y2="161"/><line x1="271" y1="121" x2="271" y2="145"/><line x1="313" y1="150" x2="313" y2="129"/><line x1="356" y1="89" x2="356" y2="113"/><line x1="398" y1="106" x2="398" y2="97"/></g><line x1="77" y1="218" x2="423" y2="87" stroke="#33312E" stroke-width="2.5"/><g fill="#185FA5"><circle cx="102" cy="194" r="5"/><circle cx="144" cy="216" r="5"/><circle cx="187" cy="158" r="5"/><circle cx="229" cy="187" r="5"/><circle cx="271" cy="121" r="5"/><circle cx="313" cy="150" r="5"/><circle cx="356" cy="89" r="5"/><circle cx="398" cy="106" r="5"/></g><text x="436" y="287" font-family="sans-serif" font-size="13" fill="#9C9B95">x</text><text x="44" y="52" font-family="sans-serif" font-size="13" fill="#9C9B95">y</text><g font-family="sans-serif" font-size="12"><circle cx="250" cy="40" r="5" fill="#185FA5"/><text x="260" y="44" fill="#0C447C">data</text><line x1="300" y1="40" x2="324" y2="40" stroke="#33312E" stroke-width="2.5"/><text x="330" y="44" fill="#33312E">fit ŷ</text><line x1="372" y1="32" x2="372" y2="48" stroke="#C0507A" stroke-width="1.8"/><text x="380" y="44" fill="#C0507A">residual</text></g></svg></div>

Why *squared* (not absolute)? It is smooth/differentiable (nice gradients), it is the
**maximum-likelihood** choice under Gaussian noise, and as a function of $w, b$ it is a
single **convex bowl** — so there is **one global minimum** and no local-minimum traps.
The cost: squaring makes it **sensitive to outliers**.

Related metrics (same idea, different units):
- **RMSE** $= \sqrt{\text{MSE}}$ — back in the units of $y$, so it's interpretable.
- **MAE** $= \frac1m\sum|\hat y_i - y_i|$ — absolute error, more robust to outliers.

## 3. The closed-form solution (ordinary least squares)

Because the MSE bowl is convex, the minimum has an **exact formula** — no iteration.

**Single feature** (worth knowing by hand):
$$ w = \frac{\operatorname{cov}(x, y)}{\operatorname{var}(x)}, \qquad b = \bar y - w\,\bar x. $$

**General case:** stack the data into a matrix $X$ (with a column of 1s for the bias).
Setting the gradient of MSE to zero gives the **normal equations**:
$$ \boldsymbol\beta = (X^\top X)^{-1} X^\top y . $$

Geometric meaning: $\hat y$ is the **orthogonal projection** of $y$ onto the column
space of $X$; the residuals come out perpendicular to that space. In the Playground,
the **Fit** button solves exactly this (via `np.linalg.lstsq`).

## 4. Gradient descent (the iterative alternative)

When $X$ is huge, streaming, or you want the method that scales to neural nets,
**descend** the MSE instead of solving it in one shot. The gradient is

$$ \nabla_{\mathbf w}\,\text{MSE} = \tfrac{2}{m}\,X^\top(\hat y - y), $$

and you step downhill:

$$ \mathbf w \leftarrow \mathbf w - \eta\,\tfrac1m X^\top(\hat y - y), \qquad b \leftarrow b - \eta\,\tfrac1m\textstyle\sum_i(\hat y_i - y_i). $$

Because the MSE is a **convex bowl**, "downhill" always leads to the one global
minimum. Each step moves $w$ against the slope (gradient); steps are **big when far
out, small near the bottom** — so it glides in and settles at the best fit $w^\*$:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 480 320" style="width:100%;max-width:470px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The mean squared error drawn as a convex bowl over the weight w; gradient descent takes steps downhill, large when far away and small near the bottom, converging at the minimum."><defs><marker id="gdah" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#9A6A2A"/></marker></defs><rect x="1" y="1" width="478" height="318" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="55" y1="275" x2="448" y2="275" stroke="#C9C8C1" stroke-width="1.2"/><line x1="55" y1="275" x2="55" y2="45" stroke="#C9C8C1" stroke-width="1.2"/><polyline points="60,142 70,154 79,166 88,177 98,187 108,196 117,204 126,212 136,219 146,226 155,231 164,236 174,240 184,243 193,246 202,248 212,249 222,249 231,249 240,248 250,246 260,244 269,240 278,236 288,232 298,226 307,220 316,213 326,205 336,197 345,188 354,178 364,167 374,156 383,143 392,130 402,117 412,103 421,87 431,72 440,55" fill="none" stroke="#5B8FC2" stroke-width="2.5"/><line x1="222" y1="249" x2="222" y2="275" stroke="#1D9E75" stroke-width="1.4" stroke-dasharray="4 3"/><text x="222" y="291" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#1D9E75">w* (best)</text><g stroke="#9A6A2A" stroke-width="1.6" fill="none"><line x1="83" y1="170" x2="116" y2="203" marker-end="url(#gdah)"/><line x1="125" y1="212" x2="146" y2="224" marker-end="url(#gdah)"/><line x1="156" y1="232" x2="168" y2="236" marker-end="url(#gdah)"/></g><g fill="#9A6A2A"><circle cx="79" cy="166" r="4.5"/><circle cx="121" cy="208" r="4.5"/><circle cx="151" cy="229" r="4.5"/><circle cx="172" cy="239" r="4.5"/><circle cx="186" cy="244" r="4.5"/><circle cx="197" cy="247" r="4.5"/><circle cx="204" cy="248" r="4.5"/></g><text x="79" y="156" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#9A6A2A">start</text><text x="436" y="293" font-family="sans-serif" font-size="13" fill="#9C9B95">w</text><text x="38" y="52" font-family="sans-serif" font-size="13" fill="#9C9B95">MSE</text><text x="352" y="86" font-family="sans-serif" font-size="12" fill="#6B6A66">convex → one minimum</text></svg></div>

- $\eta$ — the **learning rate**: too big diverges, too small crawls.
- **Scale your features first** (standardize): on very different scales the bowl
  becomes a stretched ravine and descent zig-zags slowly.

That update *is* training a linear neuron — the direct link to Tier 1 of the ANN
module. (Closed-form vs. gradient descent: exact, great for small/medium data — vs.
scalable, the basis of all deep learning.)

## 5. How good is the fit? — R² and residuals

$$ R^2 = 1 - \frac{\sum_i (\hat y_i - y_i)^2}{\sum_i (y_i - \bar y)^2} = 1 - \frac{\text{model error}}{\text{error of just predicting } \bar y}. $$

Read it as the **fraction of the variance in $y$ that the model explains**: $R^2 = 1$
is perfect, $R^2 = 0$ is no better than always guessing the mean $\bar y$, and it can
go **negative** for a genuinely bad fit. Report **RMSE** next to it for an error in
real units. And **always plot the residuals** — if they show a pattern (a curve, a
funnel widening with $x$), the straight-line assumption is wrong and you're leaving
signal on the table.

## 6. Assumptions (when OLS is "correct")

Linearity, independent errors, constant error variance (homoscedasticity), and —
for confidence intervals/tests — roughly normal residuals. Violations don't always
ruin *prediction*, but they break the *inference* (standard errors).

## 7. Regularization (preview)

With many features you overfit. **Ridge (L2)**: minimize $\text{MSE}+\lambda\lVert\mathbf w\rVert_2^2$
— shrinks weights smoothly. **Lasso (L1)**: $+\lambda\lVert\mathbf w\rVert_1$ —
drives some weights to *exactly* 0 (automatic feature selection). You'll meet L2
again as **weight decay** in the ANN module.

## 8. Multiple & polynomial regression

More features → a hyperplane. Add $x^2, x^3, \dots$ as extra features and the
*same linear method* fits curves — it's linear in the **parameters**, nonlinear in
$x$.

## 9. The ML ↔ ANN bridge

| single neuron | + loss | = classical model |
|---|---|---|
| linear $\varphi$ | MSE | **linear regression** (this lesson) |
| sigmoid $\varphi$ | cross-entropy | **logistic regression** (next ML lesson) |

The single artificial neuron you studied in the ANN module literally *is* these
classical models. The two modules meet right here.

## 10. Pitfalls

Outliers (squared loss is sensitive), multicollinearity (unstable $\boldsymbol\beta$),
extrapolating beyond the training range, and data leakage. Standardize features
when their scales differ.

## 11. Interpreting the coefficients

A fitted weight $w_j$ means: **"holding the other features fixed, a one-unit increase
in $x_j$ changes the prediction by $w_j$."** Sign = direction, magnitude = strength —
*but magnitudes are only comparable if the features share a scale*, so standardize
before comparing them. Categorical inputs (e.g. plan = A/B/C) become **one-hot / dummy**
columns; each dummy's weight is the offset versus the baseline category.

## 12. A worked example

Predict tip from bill on 4 receipts: $x = [10, 20, 30, 40]$, $y = [2, 3, 5, 6]$.
- means $\bar x = 25,\ \bar y = 4$.
- $\sum (x_i-\bar x)(y_i-\bar y) = 70$, $\;\sum (x_i-\bar x)^2 = 500$.
- slope $w = 70/500 = 0.14$; intercept $b = \bar y - w\bar x = 4 - 0.14(25) = 0.5$.

So $\hat y = 0.14\,x + 0.5$: each extra dollar of bill adds ~14¢ of tip, and a \$50
bill predicts $\hat y = 0.14(50) + 0.5 = \$7.50$. (Enter these points in the Playground
and press **Fit** to get the same line.)
"""

_TASKS = r"""
### Warm-up — in the Playground tab
1. Drag **w** and **b** to minimize "Your MSE" by hand, then press **Fit** — how
   close did your eyeball estimate get to least squares?
2. Push the **noise σ** up and watch $R^2$ fall. At σ = 0, what is $R^2$, and why?
3. Set very few points (n small) + high noise, then press Fit several times with
   different **seeds** — see how unstable the fit becomes.

### Pencil & paper
4. Derive the normal equations by setting $\partial\,\text{MSE}/\partial\boldsymbol\beta = 0$.
5. Show that for one feature, $w = \dfrac{\operatorname{cov}(x,y)}{\operatorname{var}(x)}$
   and $b = \bar y - w\bar x$.
6. Explain in one line why least squares = orthogonal projection of $y$.

### Code
7. Fit OLS two ways — `np.linalg.lstsq` **and** the normal equation
   $(X^\top X)^{-1}X^\top y$ — and check they match.
8. Implement gradient descent for linear regression; plot the loss curve and
   confirm it converges to the OLS solution.
9. Add polynomial features $[x, x^2, x^3]$ and fit a curved dataset with the same
   linear solver.
10. Add a ridge penalty $\lambda$; plot how each weight shrinks as $\lambda$ grows.

### Bridge to the ANN module
11. Once you've built the autograd engine (ANN Tier 1), train a **linear neuron**
    with MSE on the same data and confirm it converges to the OLS line.
"""

_REFERENCES = r"""
### Books (with the right chapter)
- James, Witten, Hastie & Tibshirani — *An Introduction to Statistical Learning*,
  ch. 3 — [free PDF](https://www.statlearning.com/).
- Hastie, Tibshirani & Friedman — *Elements of Statistical Learning*, ch. 3.
- Bishop — *Pattern Recognition and Machine Learning*, ch. 3 (linear regression).

### Docs & courses
- scikit-learn — [LinearRegression / Ridge / Lasso](https://scikit-learn.org/stable/modules/linear_model.html).
- Andrew Ng — *Machine Learning* (linear regression, gradient descent).
- StatQuest — linear regression & R² (intuition, on YouTube).

### In this lab
- ANN module: the single-neuron lesson (a linear neuron + MSE *is* this model).
- Next ML lessons: logistic regression (classification), then trees, SVM, clustering.
"""

_QUIZ = [
    Question(
        "Linear regression predicts what kind of target?",
        ["A discrete class label", "A continuous numeric value",
         "A probability only", "A ranking"],
        1,
        "Regression predicts a continuous value ŷ = w·x + b; predicting a class is classification.",
    ),
    Question(
        "Which quantity does ordinary least squares minimize?",
        ["Mean absolute error", "Mean squared error (MSE)",
         "Cross-entropy", "The number of misclassifications"],
        1,
        "OLS minimizes the sum/mean of squared residuals.",
    ),
    Question(
        "The normal-equation solution for the coefficients is…",
        ["XᵀX", "(XᵀX)⁻¹Xᵀy", "Xy⁻¹", "(Xy)ᵀX"],
        1,
        "β = (XᵀX)⁻¹Xᵀy — the closed-form least-squares solution.",
    ),
    Question(
        "Geometrically, least squares computes…",
        ["the median of y", "the orthogonal projection of y onto the column space of X",
         "the nearest data point", "a rotation of the axes"],
        1,
        "ŷ is y projected onto span(X); residuals are perpendicular to that space.",
    ),
    Question(
        "An R² of 0 means the model…",
        ["fits perfectly", "is no better than always predicting the mean ȳ",
         "has zero weights", "overfits"],
        1,
        "R² = 1 − SS_res/SS_tot; R²=0 means SS_res = SS_tot, i.e. no better than the mean.",
    ),
    Question(
        "Which regularizer drives some weights to exactly zero (feature selection)?",
        ["Ridge (L2)", "Lasso (L1)", "Dropout", "Batch norm"],
        1,
        "L1 (lasso) produces sparse solutions; L2 (ridge) shrinks but rarely zeroes weights.",
    ),
    Question(
        "A linear neuron trained with MSE is equivalent to…",
        ["logistic regression", "linear regression", "k-means", "a decision tree"],
        1,
        "Linear activation + MSE = linear regression — the ML↔ANN bridge.",
    ),
    Question(
        "One reason squared error is preferred over absolute error is…",
        ["it ignores outliers", "it is differentiable everywhere / matches Gaussian-noise MLE",
         "it is always smaller", "it needs no data"],
        1,
        "Squared error is smooth (nice gradients) and is the maximum-likelihood loss under Gaussian noise.",
    ),
    Question(
        "Residuals plotted vs. x show a clear U-shaped curve. This suggests…",
        ["a perfect fit", "the linear model is underfitting — add nonlinear features",
         "too much regularization", "the data is normalized"],
        1,
        "Structure in residuals means the straight line misses curvature; polynomial/nonlinear features help.",
    ),
    Question(
        "In ŷ = wx + b, the slope w represents…",
        ["the value of ŷ when x=0", "the change in ŷ per one-unit increase in x",
         "the average of y", "the error of the fit"],
        1,
        "w is the slope: how much the prediction moves for each unit of x. b is the value at x=0.",
    ),
    Question(
        "OLS has an exact closed-form solution because the MSE, as a function of the weights, is…",
        ["non-convex with many minima", "convex — a single bowl with one global minimum",
         "discontinuous", "always zero"],
        1,
        "Squared error is convex in w, b, so setting the gradient to zero gives the unique optimum (the normal equations).",
    ),
    Question(
        "RMSE is often reported instead of MSE because it…",
        ["is always smaller", "is in the same units as y, so it's interpretable",
         "ignores outliers", "needs no data"],
        1,
        "RMSE = √MSE puts the error back in the target's units (e.g. dollars), which is easier to reason about.",
    ),
]

LINEAR_REGRESSION = Lesson(
    key="linear_regression",
    title="Linear regression",
    theory=_THEORY,
    quiz=_QUIZ,
    tasks=_TASKS,
    references=_REFERENCES,
)

# ===========================================================================
# M0 — Foundations of machine learning
# ===========================================================================

_F_THEORY = r"""
## 1. What machine learning is

Machine learning **improves at a task by learning patterns from data**, instead of
being programmed rule-by-rule. The shift:

- **Traditional programming:** *you* write the rules. `rules + data → answers`.
- **Machine learning:** you supply examples of the answers, and the algorithm
  *infers* the rules. `data + answers → rules (a model)`.

Concretely — spam filtering by hand means endlessly maintaining `if "free money" in
text...`; by ML you feed thousands of labeled emails and it learns the patterns itself.

Mitchell's classic definition: a program learns from experience $E$ on task $T$
measured by $P$ if its performance at $T$ (by $P$) improves with $E$.

**The four ingredients** of every ML system (you'll see them in each lesson):
1. **Data** — examples to learn from.
2. **Model** — a family of functions with tunable **parameters** (e.g. $\hat y=\mathbf w\cdot\mathbf x+b$).
3. **Loss** — a number measuring how wrong the model is.
4. **Optimization** — a procedure that adjusts the parameters to reduce the loss.

**Reach for ML when** the rules are too many or too fuzzy to write by hand (spam,
vision, language) *and* you have data with examples of the right answer. **Don't** when
a simple rule or formula already works — ML adds complexity, data hunger, and the risk
of being confidently wrong.

## 2. The kinds of learning

- **Supervised** — learn a mapping $\mathbf x\to y$ from *labeled* examples.
  - *Regression* — continuous $y$ (house price, temperature).
  - *Classification* — discrete $y$ (spam/ham, digit 0–9, disease/healthy).

  Most of this module.
- **Unsupervised** — find structure in *unlabeled* data: *clustering* (customer
  segments), *dimensionality reduction* (PCA), *anomaly detection* (fraud).
- **Reinforcement** — an agent learns a *policy* by trial-and-error from **rewards**
  (game-playing, robotics).
- **Self-supervised** *(how modern AI is really trained)* — the labels are generated
  *from the data itself*: "predict the next word", "fill in the blank". No human
  labeling, yet it learns rich representations — this is how **LLMs and foundation
  models are pre-trained**.
- **Semi-supervised** — a little labeled data plus a lot of unlabeled data.

## 3. The ML workflow

The end-to-end pipeline. It *looks* linear but is really a **loop** — you cycle back
constantly, and most real effort goes into the *data* stages, not the modeling.

$$\text{frame}\to\text{data}\to\text{prepare}\to\text{split}\to\text{model}\to\text{train}\to\text{evaluate}\to\text{tune}\to\text{test}\to\text{deploy}\to\text{monitor}\;\circlearrowleft$$

**1. Frame the problem.** What exactly are you predicting, and is ML even the right
tool? Define the **target** $y$, whether it's regression or classification, the
**metric** you'll judge success by, and a **baseline** to beat. A clear, measurable
goal prevents most downstream confusion.

**2. Get & understand the data (EDA).** Collect data, then *look at it*: plot
distributions, check ranges, missing values, outliers, class balance, correlations.
Exploratory data analysis is where you build intuition and catch problems (and
possible leakage) **before** modeling.

**3. Prepare / preprocess.** Turn raw data into model-ready features: impute or drop
missing values, **encode** categoricals (one-hot), **scale / standardize** numeric
features, engineer new ones. Crucially, *fit these transformers on the training data
only* (see step 4) — otherwise you leak.

**4. Split — before fitting anything.** Carve out train / validation / test
(stratified for classification), or set up cross-validation. Do it **early**: any
preprocessing that "sees" the validation/test rows leaks information and inflates your
scores.

**5. Choose a model + baseline.** Start **simple** — a majority/mean baseline, then
linear or logistic regression. Add complexity (trees, ensembles, neural nets) only if
the simple model isn't enough. Simple models are fast, interpretable, and a fair
yardstick.

**6. Train (fit).** Run the learning algorithm to **minimize the loss** on the training
set — this sets the model's *parameters* (the weights $\mathbf w, b$).

**7. Evaluate.** Score on the **validation** set with the metric from step 1, always
**versus the baseline**. Then do **error analysis**: *where* does it fail and *why*?
That diagnosis drives the next loop far more than reaching for a fancier model.

**8. Tune & iterate.** Adjust **hyperparameters** (learning rate, degree, regularization
strength) via grid/random search or CV; revisit features (step 3); fight over/underfitting
(§6). Most improvement happens in this loop — and it's exactly why the test set must
stay sealed: every choice here is effectively "training" on the validation data.

**9. Final test — once.** When you've stopped choosing, evaluate **a single time** on
the untouched test set for an honest estimate of real-world performance. Peek and
re-tune, and that estimate is no longer trustworthy.

**10. Deploy.** Ship the model for inference — *batch* (score a table nightly) or
*real-time* (an API behind your app). Package the **exact** preprocessing together with
the model so production sees identical features.

**11. Monitor & maintain.** Production data drifts and performance decays
(**distribution shift**, §9). Track live metrics, detect drift, and **retrain** on
fresh data — which sends you back to step 2. The workflow never really ends.

**Three things beginners underestimate:**
- It's **iterative**, not linear — expect to loop steps 2→8 many times.
- **~80% of the work is data** (collecting, cleaning, features), not modeling.
- The **test set is sacred** — touching it during steps 5–8 is the most common way to
  fool yourself.

*In this lab:* the **Playground** above is steps 5–7 in miniature; **M1 · Regression**
is steps 6–7 with least squares; later lessons go deep on preprocessing, tuning, and
specific models.

## 4. Generalization — the whole game

The entire point of ML is **generalization**: doing well on **new, unseen** data — not
memorizing the training set. A model that scores 100% on data it has seen but flops on
fresh data has learned nothing useful (it's a lookup table, not a pattern).

- **Training error** — performance on data the model learned from (optimistic).
- **Generalization (test) error** — performance on unseen data (what you actually care about).
- The **gap** between them is the tell-tale sign of overfitting (§6).

Why a held-out **test set** estimates generalization: it stands in for "the future."
This only works under the **i.i.d. assumption** — that test data is drawn from the
*same distribution* as training data. When that breaks (the world changes), so does
your estimate — that's **distribution shift** (§9). Example: a model trained only on
summer sales will generalize poorly to the holiday season, no matter how good its test
score *looked* on summer data.

## 5. Train / validation / test split

- **Train** — fit the model's parameters.
- **Validation** — tune hyperparameters and choose between models.
- **Test** — one final, unbiased estimate (use it **once**, at the very end).

Typical split ≈ 60/20/20 (or 80/20 train/test with cross-validation inside train).
For classification use a **stratified** split so each class keeps its proportion.

**k-fold cross-validation** (when data is scarce): split the training data into $k$
folds; train on $k-1$, validate on the held-out fold, rotate $k$ times, and average
the scores. More stable than a single split and uses all data for both roles — at
$k\times$ the compute. The **test set stays untouched** until the very end.

## 6. Underfitting vs. overfitting

The central failure modes — both are *poor generalization*, for opposite reasons.
Same data, three models: too simple (misses the trend), just right (captures it), and
too flexible (chases every wiggle of the noise):

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 540 230" style="width:100%;max-width:540px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Three panels on the same data. Left: a straight line underfits. Middle: a smooth curve fits well. Right: a wiggly high-degree curve overfits by chasing the noise."><rect x="1" y="1" width="538" height="228" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g fill="#FFFFFF" stroke="#E2E2DA"><rect x="20" y="35" width="150" height="150" rx="6"/><rect x="193" y="35" width="150" height="150" rx="6"/><rect x="366" y="35" width="150" height="150" rx="6"/></g><polyline points="23,77 44,76 78,75 112,74 146,73 167,73" fill="none" stroke="#9A6A2A" stroke-width="2.5"/><polyline points="196,84 217,80 240,74 262,69 287,67 311,71 337,86 340,88" fill="none" stroke="#1D9E75" stroke-width="2.5"/><polyline points="369,52 374,88 380,92 388,78 397,70 405,72 414,75 421,75 429,72 437,68 445,69 455,73 463,76 469,74 477,66 485,59 492,64 500,87 505,99 510,80 513,35" fill="none" stroke="#C0507A" stroke-width="2.2"/><g><circle cx="28" cy="87" r="3.5" fill="#185FA5"/><circle cx="44" cy="75" r="3.5" fill="#185FA5"/><circle cx="61" cy="73" r="3.5" fill="#185FA5"/><circle cx="78" cy="74" r="3.5" fill="#185FA5"/><circle cx="95" cy="68" r="3.5" fill="#185FA5"/><circle cx="112" cy="74" r="3.5" fill="#185FA5"/><circle cx="129" cy="69" r="3.5" fill="#185FA5"/><circle cx="146" cy="63" r="3.5" fill="#185FA5"/><circle cx="162" cy="90" r="3.5" fill="#185FA5"/><circle cx="200" cy="87" r="3.5" fill="#185FA5"/><circle cx="217" cy="75" r="3.5" fill="#185FA5"/><circle cx="234" cy="73" r="3.5" fill="#185FA5"/><circle cx="251" cy="74" r="3.5" fill="#185FA5"/><circle cx="268" cy="68" r="3.5" fill="#185FA5"/><circle cx="285" cy="74" r="3.5" fill="#185FA5"/><circle cx="302" cy="69" r="3.5" fill="#185FA5"/><circle cx="319" cy="63" r="3.5" fill="#185FA5"/><circle cx="336" cy="90" r="3.5" fill="#185FA5"/><circle cx="374" cy="87" r="3.5" fill="#185FA5"/><circle cx="390" cy="75" r="3.5" fill="#185FA5"/><circle cx="407" cy="73" r="3.5" fill="#185FA5"/><circle cx="424" cy="74" r="3.5" fill="#185FA5"/><circle cx="441" cy="68" r="3.5" fill="#185FA5"/><circle cx="458" cy="74" r="3.5" fill="#185FA5"/><circle cx="475" cy="69" r="3.5" fill="#185FA5"/><circle cx="492" cy="63" r="3.5" fill="#185FA5"/><circle cx="508" cy="90" r="3.5" fill="#185FA5"/></g><g font-family="sans-serif" font-size="13" text-anchor="middle"><text x="95" y="207" fill="#9A6A2A">underfit (too simple)</text><text x="268" y="207" fill="#1D9E75">just right</text><text x="441" y="207" fill="#C0507A">overfit (too wiggly)</text></g></svg></div>

**Underfitting (high bias)** — the model is **too simple** to capture the pattern.
- *Signs:* **high train error AND high test error** (close together).
- *Causes:* model too rigid, too few features, too much regularization.
- *Fixes:* a more flexible model, more/better features, train longer, less regularization.

**Overfitting (high variance)** — the model is **too flexible** and fits the *noise*.
- *Signs:* **low train error but high test error** (a big gap).
- *Causes:* model too complex for the data, too many features, too little data.
- *Fixes:* more data, a simpler model, **regularization**, early stopping, dropout, cross-validation.

As you increase model **capacity** (e.g. polynomial degree), training error keeps
falling, but test error first falls, bottoms out at the **sweet spot**, then rises — a
**U-curve**. Crank the degree in the Playground and watch it happen. (Aside: very large
models can show a surprising *double descent* where test error falls again — beyond this
lesson.)

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 480 300" style="width:100%;max-width:470px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="As model capacity grows, training error keeps falling while test error follows a U-shape, reaching a minimum at the sweet spot before rising due to overfitting."><rect x="1" y="1" width="478" height="298" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="60" y1="250" x2="448" y2="250" stroke="#C9C8C1" stroke-width="1.2"/><line x1="60" y1="250" x2="60" y2="40" stroke="#C9C8C1" stroke-width="1.2"/><line x1="164" y1="82" x2="164" y2="250" stroke="#1D9E75" stroke-width="1.3" stroke-dasharray="4 3"/><text x="164" y="74" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#1D9E75">sweet spot</text><polyline points="60,172 95,202 129,219 164,229 198,235 233,238 267,240 302,241 336,242 371,242 405,242 440,242" fill="none" stroke="#185FA5" stroke-width="2.5"/><polyline points="60,89 95,142 129,168 164,177 198,175 233,167 267,155 302,140 336,122 371,103 405,82 440,60" fill="none" stroke="#C0507A" stroke-width="2.5"/><circle cx="164" cy="177" r="4.5" fill="#1D9E75"/><text x="96" y="116" font-family="sans-serif" font-size="11" fill="#9C9B95">underfit</text><text x="404" y="118" text-anchor="end" font-family="sans-serif" font-size="11" fill="#9C9B95">overfit</text><g font-family="sans-serif" font-size="12"><line x1="246" y1="36" x2="270" y2="36" stroke="#185FA5" stroke-width="2.5"/><text x="276" y="40" fill="#185FA5">train</text><line x1="330" y1="36" x2="354" y2="36" stroke="#C0507A" stroke-width="2.5"/><text x="360" y="40" fill="#C0507A">test</text></g><text x="252" y="278" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#6B6A66">model capacity (e.g. polynomial degree) →</text><text x="30" y="150" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#6B6A66" transform="rotate(-90 30 150)">error</text></svg></div>

## 7. The bias–variance tradeoff

The math behind §6. Imagine retraining your model on many different samples of data and
averaging the test error; it decomposes into three parts:

$$\mathbb E[\text{test error}] \approx \underbrace{\text{bias}^2}_{\text{too simple}} + \underbrace{\text{variance}}_{\text{too sensitive}} + \underbrace{\sigma^2}_{\text{irreducible noise}}.$$

- **Bias** — error from wrong assumptions; the model can't represent the truth (a line
  fitting a curve). High bias ⇒ **underfitting**.
- **Variance** — how much the fitted model swings as the training data changes; an
  over-flexible model chases noise. High variance ⇒ **overfitting**.
- **Irreducible noise** $\sigma^2$ — randomness in the data itself; no model can beat it.

The **tradeoff:** more flexibility lowers bias but raises variance, and vice-versa —
you can't drive both to zero, so you aim for the sum's minimum (the §6 sweet spot). The
levers: **more data** (lowers variance for free), **simpler/complex models** (move along
the curve), and **regularization** (trades a little bias for a lot less variance).

## 8. Evaluation: metric vs. loss

A crucial distinction: the **loss** is what the model *optimizes* (smooth,
differentiable — MSE, cross-entropy); the **metric** is what *you* report and care
about (accuracy, RMSE, F1). They are often different functions.

**Regression:** MSE / RMSE (penalize big errors), MAE (robust to outliers), $R^2$.

**Classification** starts from the **confusion matrix** (TP, FP, FN, TN):
- **Accuracy** $=\dfrac{TP+TN}{\text{all}}$ — but it *lies on imbalanced data*: if 99%
  of emails are ham, "always predict ham" scores 99% and catches **zero** spam.
- **Precision** $=\dfrac{TP}{TP+FP}$ — of those flagged positive, how many really are.
- **Recall** $=\dfrac{TP}{TP+FN}$ — of the real positives, how many we caught.
- **F1** — harmonic mean of precision & recall (good under imbalance).
- **ROC–AUC** — ranking quality across all decision thresholds.

Precision vs. recall is a **tradeoff** set by the threshold: a cancer screen wants
high *recall* (never miss a case); a spam filter wants high *precision* (never trash
real mail). Always compare to a **baseline** (predict the mean / the majority class).

## 9. Data, features & the classic pitfalls

A model is only as good as its data — "garbage in, garbage out." Often **feature
engineering** (good inputs), **scaling / standardization**, and handling missing
values & categoricals matter more than the choice of model. Watch out for:

- **Data leakage** — information from the test set or the future sneaks into training
  (e.g. standardizing using the *whole* dataset, or a feature that secretly encodes
  the answer). Gives amazing validation scores that **collapse in production** — the
  #1 way to fool yourself.
- **Class imbalance** — rare positives make accuracy misleading (§8); use F1 / resampling.
- **Distribution shift** — the world changes after training; monitor and retrain.
- **Curse of dimensionality** — in high dimensions data becomes sparse and distances
  lose meaning, so you need exponentially more data — hence dimensionality reduction
  and regularization.

## 10. Where ANN fits

Neural networks are *one family* of ML models — extremely flexible function
approximators. Everything here (generalization, the split, over/underfitting, loss,
metrics, leakage) applies to them too. The single neuron from the ANN module **is**
classic ML: linear / logistic regression.

## 11. A worked example, end-to-end

**Task:** flag customers likely to churn next month.
1. **Data & labels** — past customers with features (tenure, monthly spend, support
   tickets, plan) and a label *churned? yes/no*.
2. **Features** — standardize numeric columns; one-hot encode "plan"; drop IDs.
3. **Split** — stratified 80/20 train/test; do model selection with 5-fold CV *inside*
   the training set.
4. **Baseline** — "always predict no churn": maybe 85% accuracy but **0% recall** —
   useless, and a reminder why accuracy alone misleads.
5. **Model** — logistic regression (a sigmoid neuron!). Train; pick the decision
   threshold from the precision/recall tradeoff (churn prevention favors **recall**).
6. **Evaluate** — on the untouched test set: precision, recall, F1, ROC-AUC vs. the
   baseline.
7. **Iterate** — add features, regularize if overfitting, rebalance classes.
8. **Deploy & monitor** — watch for distribution shift; retrain on fresh data.

Every later ML lesson is a deeper dive into one of these steps.

## 12. Key concepts & vocabulary

- **Parameters vs. hyperparameters** — *parameters* are learned from data (the weights
  $\mathbf w, b$); *hyperparameters* are set by you (learning rate, polynomial degree,
  regularization strength, $k$). Hyperparameters are chosen on the **validation** set.
- **Loss vs. metric** — what's optimized vs. what's reported (§8).
- **Inductive bias** — the assumptions a model builds in (linearity, smoothness,
  locality). With *no* assumptions there is no generalization.
- **No Free Lunch theorem** — no single model is best on *all* problems; the right
  choice depends on the data. Start with simple baselines.
- **Regularization** — penalize complexity (L1/L2, dropout, early stopping) to fight
  overfitting — the main lever on the bias–variance dial.
- **Reproducibility** — fix random **seeds** so splits and results repeat.
"""

_F_TASKS = r"""
### Warm-up — in the Playground tab
1. Set the polynomial **degree to 1** — describe the fit. Now set it to **12**. Which
   one has low *train* error but bad *test* error? Name each failure (under/overfit).
2. Find the degree that **minimizes test error**. Does it change when you raise the
   **noise**?
3. Shrink the **train fraction** — what happens to overfitting and to the test error?

### Pencil & paper
4. For each, name the ML type: spam detection; grouping customers; a game-playing bot;
   predicting tomorrow's temperature.
5. Explain in one line why the test set must be touched only once.
6. Write the bias–variance decomposition and say which term regularization reduces.

### Code
7. Implement a train/test split and **k-fold cross-validation** (NumPy or
   `sklearn.model_selection`).
8. Reproduce the Playground's U-curve: plot **train vs. test MSE** as polynomial
   degree goes 1→12.
9. Plot a **learning curve** (error vs. training-set size) for a fixed model.

### Bridge
10. The ANN module's neuron + MSE is linear regression (degree 1 here). Confirm a
    degree-1 fit matches the ML module's least-squares line on the same data.
"""

_F_REFERENCES = r"""
### Books
- James, Witten, Hastie & Tibshirani — *Introduction to Statistical Learning*,
  ch. 2 (statistical learning) & ch. 5 (resampling) — [free PDF](https://www.statlearning.com/).
- Mitchell — *Machine Learning* (the classic definition of learning).

### Courses & docs
- Google — [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course).
- scikit-learn — [cross-validation & model selection](https://scikit-learn.org/stable/modules/cross_validation.html).
- Andrew Ng — *Machine Learning* (bias/variance, train/dev/test).

### In this lab
- ML module: M1 regression builds directly on these foundations.
- ANN module: generalization, loss, and over/underfitting apply identically to nets.
"""

_F_QUIZ = [
    Question(
        "Machine learning differs from traditional programming because it…",
        ["needs no data", "learns patterns from data instead of following hand-written rules",
         "is always faster", "never makes errors"],
        1,
        "ML infers the rules from examples; you supply data + a goal, not explicit if/else logic.",
    ),
    Question(
        "Predicting house price from labeled examples is which kind of ML?",
        ["unsupervised", "supervised regression", "reinforcement learning", "clustering"],
        1,
        "Labeled examples + a continuous target = supervised regression.",
    ),
    Question(
        "Grouping customers into segments with no labels is…",
        ["supervised classification", "unsupervised (clustering)", "regression", "reinforcement learning"],
        1,
        "No labels, find structure → unsupervised learning (clustering).",
    ),
    Question(
        "The test set should be used…",
        ["throughout training to guide it", "once, for a final unbiased estimate",
         "to pick hyperparameters", "to add more training data"],
        1,
        "Touching the test set during development leaks information and inflates your score.",
    ),
    Question(
        "Low training error but high test error indicates…",
        ["underfitting", "overfitting", "high bias", "data leakage into the test set"],
        1,
        "The model memorized the training data (including noise) and fails to generalize → overfitting (high variance).",
    ),
    Question(
        "A model that is too simple (high bias) shows…",
        ["low train and low test error", "high train AND high test error",
         "low train but high test error", "perfect accuracy"],
        1,
        "Underfitting: it can't even fit the training data, so both errors are high.",
    ),
    Question(
        "Increasing model flexibility generally…",
        ["raises bias, lowers variance", "lowers bias, raises variance",
         "lowers both", "changes neither"],
        1,
        "The bias–variance tradeoff: more flexible models fit better (less bias) but are more sensitive (more variance).",
    ),
    Question(
        "Data leakage is when…",
        ["the dataset is too small", "test/future information contaminates training, giving over-optimistic scores",
         "you use too many features", "the model trains too long"],
        1,
        "Leakage makes validation look great but performance collapses in production.",
    ),
    Question(
        "LLMs and most foundation models are pre-trained mainly via…",
        ["supervised learning on human labels", "self-supervised learning (e.g. predict the next word)",
         "reinforcement learning only", "unsupervised clustering"],
        1,
        "Self-supervised: labels come from the data itself (next-token / masked-token prediction) — no human labeling needed.",
    ),
    Question(
        "k-fold cross-validation is mainly used to…",
        ["replace the test set", "get a more stable performance estimate (and use all data) when data is scarce",
         "speed up training", "increase the dataset size"],
        1,
        "Rotate which fold is held out and average — a steadier estimate than one split. The test set still stays untouched.",
    ),
    Question(
        "A spam filter that must never put real email in the spam folder should prioritize…",
        ["recall", "precision", "accuracy", "training speed"],
        1,
        "High precision = few false positives = real mail isn't trashed. (A cancer screen would instead favor recall.)",
    ),
    Question(
        "The learning rate, polynomial degree, and regularization strength are…",
        ["parameters learned from data", "hyperparameters you set (chosen on the validation set)",
         "evaluation metrics", "always fixed at 1"],
        1,
        "Parameters (w, b) are learned; hyperparameters are configured by you and tuned on validation data.",
    ),
    Question(
        "In a typical real-world ML project, the largest share of effort goes into…",
        ["choosing the fanciest model", "collecting, cleaning, and preparing the data",
         "buying GPUs", "writing the deployment API"],
        1,
        "Roughly 80% is data work (EDA, cleaning, feature engineering); modeling is the smaller, later part.",
    ),
]

FOUNDATIONS = Lesson(
    key="ml_foundations",
    title="Foundations of ML",
    theory=_F_THEORY,
    quiz=_F_QUIZ,
    tasks=_F_TASKS,
    references=_F_REFERENCES,
)

# ===========================================================================
# M2 — Classification (logistic regression)
# ===========================================================================

_C_THEORY = r"""
## 1. What classification does

Predict a **discrete class** (a category) rather than a number. Three flavors:
- **Binary** — two classes (spam/ham, fraud/legit, disease/healthy). The core case below.
- **Multiclass** — one label out of many, mutually exclusive (digit 0–9, species).
- **Multi-label** — several labels at once (an image tagged *beach* + *sunset* + *dog*).

The model carves feature space into regions separated by a **decision boundary**, and a
point is classified by which side it falls on. Crucially, most classifiers first output
a **score or probability** per class, and a **threshold** converts that to the final
label (§5) — which is exactly what lets you tune the precision/recall balance.

## 2. Why not just use linear regression?

Fitting a line to 0/1 labels and thresholding works badly: predictions are unbounded
(you get 1.7 or −0.3, not probabilities), and one far-away point can swing the line and
flip many decisions. We want an output in $[0,1]$ readable as a **probability**. Enter
logistic regression.

## 3. Logistic regression — the sigmoid neuron

Squash the linear score through a **sigmoid**:

$$ p = P(y{=}1\mid\mathbf x) = \sigma(z) = \frac{1}{1+e^{-z}}, \qquad z = \mathbf w\cdot\mathbf x + b. $$

The sigmoid is an **S-curve**: near $z=0$ it's roughly linear, and it **saturates**
toward 0 and 1 in the tails — so the model is least certain near the boundary and very
certain far from it. The output $p\in(0,1)$ reads directly as the probability of the
positive class.

**The log-odds view (why it's *logistic*).** Invert the sigmoid:
$$ z = \ln\frac{p}{1-p} = \text{log-odds}. $$
So the linear part $z$ *is* the log-odds, and a weight $w_j$ means: **a one-unit increase
in $x_j$ adds $w_j$ to the log-odds**, i.e. multiplies the **odds** by $e^{w_j}$ (the
*odds ratio*). Example: $w_j = 0.7 \Rightarrow e^{0.7}\approx 2$, so that feature roughly
**doubles the odds** of the positive class per unit. This interpretability is why logistic
regression is everywhere in medicine, credit scoring, and the social sciences.

**This is exactly the single neuron from the ANN module with a sigmoid activation** —
classification *is* one neuron.

## 4. The loss — cross-entropy (log loss)

Train by minimizing **binary cross-entropy** (a.k.a. log loss) — the negative
log-likelihood of the data under the model:

$$ \text{BCE} = -\frac1m\sum_i\big[\,y_i\log p_i + (1-y_i)\log(1-p_i)\,\big]. $$

Read one example: when the truth is $y=1$ the penalty is $-\log p$ — it is **0 when
$p=1$** (confident and right) and **$\to\infty$ as $p\to0$** (confident and *wrong*).
So cross-entropy punishes confident mistakes brutally, which pushes the model toward
honest probabilities. (Predicting $p=0.01$ for a true positive costs $-\log 0.01\approx
4.6$; predicting $p=0.9$ costs only $\approx0.1$.)

Why not MSE? With a sigmoid, MSE is **non-convex** and its gradient nearly vanishes when
the neuron is confidently wrong (slow learning). BCE is convex in $\mathbf w, b$ and
gives the clean gradient $\partial L/\partial z = p - y$ — the awkward $\sigma'$ cancels
(you measure this in ANN experiment e08).

## 5. The decision boundary & the threshold

Classify positive when $p \ge t$ (default **threshold** $t = 0.5$, i.e. $z \ge 0$).
Since $z$ is linear, the boundary $\mathbf w\cdot\mathbf x + b = 0$ is a **straight line
/ hyperplane** — logistic regression is a *linear* classifier (same shape as the neuron's
gates). Moving $t$ changes how eagerly you say "positive": lower $t$ ⇒ more positives ⇒
**higher recall, lower precision** (§7). The Playground's threshold slider shows it live.

## 6. Training

No closed form (unlike linear regression) — use **gradient descent** on BCE. It is
*literally* training a sigmoid neuron with cross-entropy. Standardize features for
faster, stabler convergence.

## 7. Evaluating a classifier

At a chosen threshold every prediction is a true/false positive/negative; tally them in
the **confusion matrix** (TP, FP, FN, TN), then:

| metric | formula | reads as |
|---|---|---|
| accuracy | $(TP+TN)/\text{all}$ | overall correct — **misleads if imbalanced** |
| precision | $TP/(TP+FP)$ | of predicted-positives, how many are right |
| recall (TPR) | $TP/(TP+FN)$ | of actual positives, how many we caught |
| F1 | $2PR/(P+R)$ | balance of precision & recall |

**Worked example.** Of 100 emails (50 spam) a filter gives TP=40, FN=10, FP=5, TN=45:
accuracy $=85\%$, precision $=40/45\approx0.89$, recall $=40/50=0.80$, F1 $\approx0.84$.
Missing 10 spam (recall 0.80) vs. trashing 5 real mails (precision 0.89) — the threshold
decides which error you prefer.

**ROC & AUC** sweep *every* threshold: plot **TPR (recall)** vs. **FPR** $=FP/(FP+TN)$.
A perfect classifier hugs the top-left corner; the diagonal is random guessing. The
**AUC** (area under the curve) is one threshold-free number = the probability the model
scores a random positive above a random negative (0.5 = chance, 1.0 = perfect) — the
go-to summary under class imbalance.

**Which metric?** Match it to the cost of each error: medical screening → **recall**
(don't miss disease); spam / recommendations → **precision** (don't annoy users); ranking
→ **AUC**. Always beat a **baseline** (majority class).

## 8. More than two classes

- **Softmax regression** — a softmax output gives a probability per class summing to 1,
  trained with cross-entropy. This is the **same softmax** an LLM uses over its vocabulary.
- **One-vs-rest** — train one binary classifier per class, pick the most confident.

## 9. Other classifiers (the M2+ map)

Logistic regression is the linear baseline. Also: **k-NN** (vote of nearest neighbors),
**trees / random forests / boosting** (M3), **SVM** (max-margin + kernels, M4), **naive
Bayes**. Start linear; reach for these when the boundary is non-linear.

## 10. The ML ↔ ANN bridge

| single neuron | + loss | = classical model |
|---|---|---|
| sigmoid $\varphi$ | binary cross-entropy | **logistic regression** (this lesson) |
| softmax | cross-entropy | **softmax / multinomial regression** |

Binary or multiclass, a linear classifier is exactly a **one-neuron (one-layer) net**.
Stack layers and the boundary becomes non-linear — the doorway to the ANN module.

## 11. A worked example, end-to-end

**Predict diabetes** from `[glucose, BMI, age]`.
1. **Data / labels** — patient records, label *diabetic? 0/1*; **standardize** the 3 features.
2. **Split** — stratified 80/20 (keep the positive rate in both halves).
3. **Baseline** — "always predict healthy": high accuracy if diabetes is rare, but
   **recall 0** — useless, and a reminder why accuracy alone misleads.
4. **Model** — logistic regression (a sigmoid neuron), trained with cross-entropy via
   gradient descent. Coefficients are interpretable: $e^{w_\text{glucose}}$ is the odds
   ratio per standardized unit of glucose.
5. **Threshold** — screening favors **recall**, so set $t$ below 0.5 to catch more cases
   (accepting more false positives → harmless follow-up tests).
6. **Evaluate** — precision / recall / F1 and **AUC** on the held-out set, vs. the baseline.
7. **Iterate & deploy** — add features, regularize, rebalance; then monitor for drift.

## 12. Probabilities & calibration

A classifier's output is only trustworthy as a probability if it is **calibrated** —
when it says "70%", the event should happen ~70% of the time. Logistic regression
trained with cross-entropy tends to be reasonably calibrated; many models (boosted
trees, SVMs, deep nets) are **over-confident** and need post-hoc fixing (Platt scaling,
isotonic regression). Check with a **reliability diagram** (predicted vs. observed
frequency). Calibration matters whenever you *act on the number itself* —
expected-value decisions, cost-based thresholds, or ranking by risk.
"""

_C_TASKS = r"""
### Warm-up — in the Playground tab
1. Slide the **threshold** from 0.5 down to 0.2 and up to 0.8. Watch the confusion matrix,
   **precision**, and **recall** trade off. Which direction helps a spam filter? a cancer screen?
2. Increase the **overlap** between classes — does accuracy drop? Is the boundary still straight?
3. Find the threshold that **maximizes F1**.

### Pencil & paper
4. From confusion matrix TP=40, FN=10, FP=5, TN=45, compute accuracy, precision, recall, F1.
5. Explain why thresholding a *linear regression* fit is fragile on 0/1 labels.
6. Derive $\partial\,\text{BCE}/\partial z = p - y$ for a sigmoid output.

### Code
7. Implement logistic regression with gradient descent in NumPy; compare its boundary to the Playground.
8. Plot an **ROC curve** by sweeping the threshold 0→1 and plotting (FPR, TPR); compute AUC.
9. Try `sklearn.linear_model.LogisticRegression` on the same data and compare.

### Bridge to ANN
10. A sigmoid neuron + BCE *is* this model. After ANN Tier 1, train that neuron on the same
    two-class data and confirm you get the same boundary.
"""

_C_REFERENCES = r"""
### Books
- James, Witten, Hastie & Tibshirani — *Intro to Statistical Learning*, ch. 4 — [free PDF](https://www.statlearning.com/).
- Bishop — *Pattern Recognition and ML*, ch. 4 (linear classification, logistic regression).

### Docs & courses
- scikit-learn — [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
  & [metrics / ROC](https://scikit-learn.org/stable/modules/model_evaluation.html).
- Google ML Crash Course — classification, ROC & AUC, precision/recall.

### In this lab
- ANN module: this is the **sigmoid neuron**; ANN e08 derives the cross-entropy gradient.
- ML module: M0 §8 (metrics), M1 (its regression cousin).
"""

_C_QUIZ = [
    Question(
        "Classification predicts…",
        ["a continuous number", "a discrete class / category", "a probability density", "a ranking only"],
        1,
        "Classification outputs a category; predicting a continuous value is regression.",
    ),
    Question(
        "The output of logistic regression is…",
        ["any real number", "a probability in (0,1) via the sigmoid", "always 0 or 1 exactly", "the log-odds directly as the answer"],
        1,
        "σ(z) maps the linear score to a probability in (0,1); you threshold it to decide the class.",
    ),
    Question(
        "Logistic regression is trained by minimizing…",
        ["mean squared error", "binary cross-entropy (log loss)", "accuracy", "the R² score"],
        1,
        "Cross-entropy is convex with a sigmoid and gives the clean gradient p−y; MSE is non-convex here.",
    ),
    Question(
        "The decision boundary of logistic regression is…",
        ["a circle", "linear — a line / hyperplane (w·x+b=0)", "always non-linear", "a single point"],
        1,
        "z = w·x+b is linear, so the boundary where p=0.5 (z=0) is a straight line / hyperplane.",
    ),
    Question(
        "Lowering the decision threshold from 0.5 to 0.2…",
        ["predicts fewer positives", "predicts more positives → higher recall, lower precision",
         "has no effect", "improves both precision and recall"],
        1,
        "A lower bar for 'positive' catches more true positives (recall ↑) but also more false positives (precision ↓).",
    ),
    Question(
        "A sigmoid neuron trained with cross-entropy is exactly…",
        ["linear regression", "logistic regression", "k-means", "a decision tree"],
        1,
        "Sigmoid activation + BCE = logistic regression — the ML↔ANN bridge.",
    ),
    Question(
        "For mutually-exclusive multiclass classification, the output layer uses…",
        ["one sigmoid", "softmax (one probability per class, summing to 1)", "ReLU", "linear"],
        1,
        "Softmax gives a probability distribution over classes — the same operation an LLM uses over its vocabulary.",
    ),
    Question(
        "Why is accuracy a poor metric when one class is rare?",
        ["it is hard to compute", "a trivial majority-class predictor scores high while catching no positives",
         "it needs probabilities", "it only works for regression"],
        1,
        "On 99% negatives, 'always predict negative' is 99% accurate but useless — use precision/recall/F1.",
    ),
    Question(
        "A logistic-regression weight w_j = 0.7 means a one-unit increase in x_j multiplies the odds by about…",
        ["0.7", "e^0.7 ≈ 2 (doubles the odds)", "0.5", "7"],
        1,
        "z is the log-odds, so adding w_j to it multiplies the odds by e^{w_j}. e^0.7 ≈ 2.",
    ),
    Question(
        "Under cross-entropy, predicting p = 0.01 for a true positive incurs…",
        ["zero loss", "a very large loss (−log 0.01 ≈ 4.6)", "a small loss", "an undefined loss"],
        1,
        "Cross-entropy punishes confident-wrong predictions harshly; the penalty →∞ as p→0 for a true positive.",
    ),
    Question(
        "ROC-AUC measures…",
        ["accuracy at threshold 0.5", "ranking quality — the chance a random positive scores above a random negative",
         "the training loss", "the number of features"],
        1,
        "AUC is threshold-free: 0.5 = random, 1.0 = perfect ranking. Handy under class imbalance.",
    ),
    Question(
        "A classifier is 'well-calibrated' when…",
        ["it has 100% accuracy", "among cases it labels '70% likely', about 70% are actually positive",
         "it uses a softmax", "its weights are all positive"],
        1,
        "Calibration = predicted probabilities match observed frequencies; check with a reliability diagram.",
    ),
]

CLASSIFICATION = Lesson(
    key="classification",
    title="Classification (logistic regression)",
    theory=_C_THEORY,
    quiz=_C_QUIZ,
    tasks=_C_TASKS,
    references=_C_REFERENCES,
)

REGISTRY = {
    FOUNDATIONS.key: FOUNDATIONS,
    LINEAR_REGRESSION.key: LINEAR_REGRESSION,
    CLASSIFICATION.key: CLASSIFICATION,
}
