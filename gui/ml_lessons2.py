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

Same machinery, continuous target. The leaf predicts the **mean** of its training
$y$'s, and splits are chosen to minimize **variance / MSE** in the children instead of
Gini. The prediction surface is a **piecewise-constant staircase** — flexible, but it
can't extrapolate beyond the training range.

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

A **Support Vector Machine** is a linear classifier with a special goal: among all
boundaries that separate the classes, choose the one with the **largest margin** — the
widest empty "street" between the two classes. The intuition: a boundary that stays far
from every point is the most robust to noise and generalizes best.

## 2. The margin, made precise

Put the boundary at $\mathbf w\cdot\mathbf x + b = 0$ and scale $(\mathbf w,b)$ so the
closest points satisfy $y_i(\mathbf w\cdot\mathbf x_i+b) = 1$. Then the **street's width**
is $\dfrac{2}{\lVert\mathbf w\rVert}$. Maximizing the margin is therefore

$$ \min_{\mathbf w,b}\ \tfrac12\lVert\mathbf w\rVert^2 \quad\text{s.t.}\quad y_i(\mathbf w\cdot\mathbf x_i+b)\ge 1\ \forall i. $$

A clean convex (quadratic) program with a unique solution.

## 3. Support vectors

At the optimum, only the points **on the margin** ($y_i(\mathbf w\cdot\mathbf x_i+b)=1$)
have nonzero influence — these are the **support vectors**. Move or delete any *other*
point and the boundary doesn't change. Consequences: the model is **sparse** (defined by
a few points), memory-light at prediction time, and robust to far-away data.

## 4. Soft margin & the C knob

Real data overlaps, so allow **slack** $\xi_i\ge0$ (a point may sit inside the margin or
be misclassified):
$$ \min\ \tfrac12\lVert\mathbf w\rVert^2 + C\sum_i\xi_i \quad\text{s.t.}\quad y_i(\mathbf w\cdot\mathbf x_i+b)\ge 1-\xi_i. $$
**C** is the regularization dial (the M0 bias–variance knob):
- **large C** → punish violations hard → narrow margin, fits training data (low bias, high variance);
- **small C** → tolerate violations → wide margin, smoother (high bias, low variance).

## 5. The hinge-loss view

That soft-margin program is equivalent to minimizing the **hinge loss** plus L2:
$$ \sum_i \max\!\big(0,\ 1 - y_i\,f(\mathbf x_i)\big) + \lambda\lVert\mathbf w\rVert^2,\qquad \lambda = \tfrac{1}{2C}. $$
Hinge loss is **0 once a point is correct *and* beyond the margin** — so only border
cases drive the fit. Contrast logistic regression's **log loss**, which every point
nudges and which yields probabilities. (Both are linear classifiers; they differ only in
the loss.)

## 6. The kernel trick — non-linear boundaries for free

The whole optimization (its dual form) touches the data only through **dot products**
$\mathbf x_i\cdot\mathbf x_j$. Replace each with a **kernel** $K(\mathbf x_i,\mathbf x_j)$
— the dot product in some high-dimensional feature space $\phi(\mathbf x)$ — and you get
a **non-linear boundary in the original space without ever computing $\phi$**. Any
$K$ satisfying Mercer's condition (it must correspond to a valid inner product) works.

## 7. Common kernels

- **Linear** $K=\mathbf x\cdot\mathbf x'$ — the plain max-margin line; great for
  high-dimensional/sparse data (text/TF-IDF).
- **Polynomial** $K=(\mathbf x\cdot\mathbf x' + c)^d$ — degree-$d$ curved boundaries.
- **RBF / Gaussian** $K=e^{-\gamma\lVert\mathbf x-\mathbf x'\rVert^2}$ — the workhorse.
  It's a **similarity** that decays with distance, mapping to an *infinite*-dimensional
  space. **gamma** sets each point's reach: large γ → tight, wiggly boundary (overfit);
  small γ → smooth, almost linear.

## 8. Choosing C and γ (and scaling!)

C and γ interact and are chosen together by cross-validated **grid search** (M6) — a
classic 2-D grid over log-spaced values. **Always standardize features first:** the RBF
kernel uses Euclidean distance, so unscaled features silently dominate. This sensitivity
to scaling is the #1 SVM gotcha.

## 9. Multiclass & regression

SVMs are binary at heart. For $>2$ classes, libraries train **one-vs-one** or
**one-vs-rest** and vote. The margin idea also gives **Support Vector Regression
(SVR)**: fit a tube of half-width $\varepsilon$ around the data and penalize only points
outside it.

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

No labels — only inputs $\mathbf x$. The goal is to **find structure**: group similar
points (clustering), compress to fewer dimensions (dimensionality reduction), or flag
the unusual (anomaly detection). There's no "accuracy" because there's no ground truth —
which makes *evaluation* the hard part (§9).

## 2. k-means — objective & algorithm

Partition points into **k** clusters, each summarized by a **centroid** $\mu_j$.
k-means minimizes the **inertia** (within-cluster sum of squares):
$$ J = \sum_{i} \lVert \mathbf x_i - \mu_{c(i)}\rVert^2 . $$
**Lloyd's algorithm** is coordinate descent on $J$, alternating:
1. **Assign** each point to its nearest centroid (fix $\mu$, optimize assignments).
2. **Update** each centroid to the mean of its points (fix assignments, optimize $\mu$).

Each step can only lower $J$, so it **converges — but to a local minimum**, not
necessarily the best one.

## 3. Initialization matters — k-means++ & restarts

Because it finds a *local* optimum, the starting centroids matter. **k-means++** spreads
the initial centroids out (each new seed chosen far from existing ones), giving much
better, more consistent results than random init. In practice you also run several
**restarts** and keep the lowest-inertia solution. Watch the seed change the outcome in
the Playground.

## 4. Choosing k

No label tells you the "right" k. Heuristics:
- **Elbow method** — plot inertia vs. k; pick the "elbow" where extra clusters stop
  helping much. (Inertia *always* falls as k rises, so you can't just minimize it.)
- **Silhouette score** — for a point, let $a$ = mean distance to its own cluster and
  $b$ = mean distance to the nearest *other* cluster; its silhouette is
  $s = \dfrac{b-a}{\max(a,b)} \in [-1,1]$. Average over points; higher = better-separated.
- **Gap statistic**, or just let a **downstream task** decide.

## 5. k-means limitations

It assumes **spherical, similar-size** clusters and uses Euclidean distance, so it
struggles with **elongated, uneven, or non-convex** clusters, is sensitive to outliers
and to **feature scale** (standardize first!), and needs **k fixed in advance**. When
these bite, reach for the alternatives below.

## 6. Hierarchical clustering

Builds a whole tree (**dendrogram**) of nested clusters. **Agglomerative**: start with
each point as its own cluster, then repeatedly **merge the two closest** clusters under a
*linkage* rule — **single** (nearest pair, makes chains), **complete** (farthest pair,
compact), **average**, or **Ward** (minimize variance increase, k-means-like). Cut the
tree at any height to read off that many clusters — **no need to fix k up front**.

## 7. Density-based clustering (DBSCAN)

Defines clusters as **dense regions** separated by sparse gaps. Points are **core**
(enough neighbors within radius `eps`), **border**, or **noise/outliers**. DBSCAN finds
**arbitrarily shaped** clusters and **doesn't need k**, but is sensitive to `eps` /
`min_samples` and struggles when densities vary.

## 8. Soft clustering — Gaussian Mixture Models

A **GMM** models the data as a mixture of Gaussians and gives each point a **soft**
membership (responsibility) per cluster, fit by **Expectation–Maximization (EM)** —
alternate "estimate responsibilities" and "update Gaussian parameters". k-means is the
hard-assignment, equal-spherical-covariance limit of a GMM.

## 9. Dimensionality reduction — why

High dimensions hurt: the **curse of dimensionality** (data gets sparse, distances lose
meaning), you can't **visualize** beyond 3-D, and storage/compute balloon. Reduction
finds a smaller set of features that keeps most of the information — for visualization,
compression, **denoising**, and as preprocessing for other models.

## 10. PCA, made precise

The workhorse linear method. Center the data, form the covariance matrix
$C = \tfrac1n X^\top X$; its **eigenvectors** are the **principal components** (the
orthogonal directions of maximum variance), ordered by eigenvalue $\lambda_i$.
Equivalently, take the **SVD** $X = U\Sigma V^\top$ — the columns of $V$ are the
components. Project onto the top few. The **explained-variance ratio**
$\lambda_i / \sum_j \lambda_j$ says how much information each keeps; choose enough
components to reach (say) 95% cumulative variance (read it off a **scree plot**).
Standardize first so large-scale features don't dominate. *(This is the Math module's
eigen/SVD/projection in action.)*

## 11. Non-linear embeddings — t-SNE & UMAP

For **visualization**, t-SNE and UMAP map high-dim data to 2-D while preserving *local*
neighborhoods, revealing clusters PCA might hide. Caveats: cluster **sizes and distances
between clusters aren't meaningful**, results vary with hyperparameters/seed, and they're
for *looking*, not as model features.

## 12. Anomaly detection

Flag points that don't fit the learned structure: far from any centroid, in a
low-density region (DBSCAN noise), low GMM likelihood, or via dedicated methods
(Isolation Forest, one-class SVM). Used heavily for fraud, fault, and intrusion detection.

## 13. Evaluating without labels

No accuracy. Use **internal** indices (inertia, **silhouette**, Davies–Bouldin),
**stability** across runs/subsamples, or — most honestly — the **downstream task** the
structure feeds (do the segments improve targeting? do the PCA features help the
classifier?). Usefulness, not a single number, is the real test.

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
