"""Math module lesson content. Reuses the Lesson/Question/quiz engine from lessons.py."""

from __future__ import annotations

from lessons import Lesson, Question

_THEORY = r"""
## 1. What a vector is

An ordered list of numbers, $\mathbf x=(x_1,\dots,x_n)\in\mathbb R^n$. Two readings:
a **point** in $n$-dimensional space, or an **arrow** from the origin. In ML
*everything* is a vector — a data sample (its features), a neuron's weights, a
word's embedding.

## 2. Basic operations

- **Addition** $(\mathbf a+\mathbf b)_i=a_i+b_i$ — arrows tip-to-tail.
- **Scalar multiply** $c\mathbf a$ — scales length ($c<0$ flips direction).
- **Magnitude (L2 norm)** $\lVert\mathbf a\rVert=\sqrt{\sum_i a_i^2}$ — the arrow's length.
- **Unit vector** $\hat{\mathbf a}=\mathbf a/\lVert\mathbf a\rVert$ — direction only.

## 3. The dot product — the star of the show

Two equivalent definitions:

$$ \mathbf a\cdot\mathbf b=\sum_i a_i b_i=\lVert\mathbf a\rVert\,\lVert\mathbf b\rVert\cos\theta. $$

It measures **alignment**: large positive when the vectors point the same way, $0$
when perpendicular, negative when opposed.

- **Cosine similarity** $\cos\theta=\dfrac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\lVert\mathbf b\rVert}$
  — similarity of *direction*, independent of length.
- **Projection** of $\mathbf b$ onto $\mathbf a$ has length $\dfrac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert}$
  — "how much of $\mathbf b$ lies along $\mathbf a$".
- **Orthogonal** $\iff \mathbf a\cdot\mathbf b=0$.

## 4. Why this is the foundation of ML & ANN

- A neuron's pre-activation **is** a dot product: $z=\mathbf w\cdot\mathbf x+b$. The
  neuron responds most when the input *aligns* with its weights — the "matched
  filter" view from the ANN neuron lesson (§3).
- **Embeddings, search & RAG:** words/documents become vectors; relevance =
  cosine similarity; "find similar" = nearest dot products.
- **Attention** in Transformers scores tokens with dot products (query · key).
- Norms power **regularization** (L1/L2) and **normalization**.

## 5. From vectors to matrices

A whole layer applies many dot products at once: $\mathbf z = W\mathbf x + \mathbf b$,
where each **row of $W$ is one neuron's weight vector**. Matrix multiplication is a
batch of dot products — the single most-executed operation in deep learning, and
exactly what GPUs accelerate.

## 6. Norms you'll meet

- $\lVert\mathbf x\rVert_2=\sqrt{\sum_i x_i^2}$ (Euclidean) → ridge / weight decay.
- $\lVert\mathbf x\rVert_1=\sum_i|x_i|$ (Manhattan) → lasso / sparsity.

These are the same L2 and L1 that show up as regularizers in the ML and ANN modules.

## 7. Matrices as transformations

A matrix $W$ is a **linear transformation**: $W\mathbf x$ rotates, scales, and shears
$\mathbf x$. A neural-net layer $W\mathbf x + \mathbf b$ is exactly an **affine
transformation** of its input, and **matrix multiplication composes transformations**
(and, as in §5, is a batch of dot products). The **identity** leaves vectors unchanged;
the **transpose** $W^\top$ shows up in backprop, where gradients flow *backward* through
$W^\top$.

## 8. Eigenvectors & SVD (a peek)

Some directions are special: an **eigenvector** $\mathbf v$ of a (square) matrix is only
**scaled**, not rotated — $W\mathbf v = \lambda\mathbf v$, with eigenvalue $\lambda$. The
**singular value decomposition** $X = U\Sigma V^\top$ generalizes this to *any* matrix
and is the engine behind **PCA** (M5): the top singular directions capture the most
variance in the data. You'll meet these again in dimensionality reduction — this is where
linear algebra pays off for ML.
"""

_TASKS = r"""
### Warm-up — in the Playground tab
1. Rotate $\mathbf x$ until it points the **same way** as $\mathbf w$ — what happens
   to the dot product and to $\theta$? Now make them **perpendicular**.
2. Point $\mathbf x$ **opposite** to $\mathbf w$ — what sign is the dot product?
3. Double the length of $\mathbf x$ — does $\cos\theta$ change? Does $\mathbf w\cdot\mathbf x$?

### Pencil & paper
4. Compute $\lVert(3,4)\rVert$, then $\cos\theta$ between $(1,0)$ and $(1,1)$.
5. Compute $(1,2,3)\cdot(0,1,0)$ and explain why it "selects" the 2nd component.
6. Show that $\mathbf a\cdot\mathbf b=0$ means the arrows are perpendicular.

### Code
7. Implement `dot`, `norm`, and `cosine_similarity` in NumPy; check against `np.dot`.
8. Represent two short sentences as bag-of-words vectors and rank them by cosine
   similarity to a query — a 10-line search engine.
9. Verify $\mathbf w\cdot\mathbf x=\lVert\mathbf w\rVert\lVert\mathbf x\rVert\cos\theta$
   numerically for random vectors.

### Bridge to ANN / ML
10. Connect it: a neuron computes $z=\mathbf w\cdot\mathbf x+b$. Build a length-$n$
    weight vector and confirm the Playground neuron's $z$ equals your hand-computed
    dot product plus the bias.
"""

_REFERENCES = r"""
### Books
- Deisenroth, Faisal & Ong — *Mathematics for Machine Learning* — [free PDF](https://mml-book.github.io/) (the single best fit for this module).
- Strang — *Introduction to Linear Algebra* (the classic).

### Video & interactive
- **3Blue1Brown** — [Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) (vectors, dot products, matrices — start here).
- Khan Academy — linear algebra (vectors & dot products).

### In this lab
- ANN module: the neuron's pre-activation $z=\mathbf w\cdot\mathbf x+b$ is this dot product.
- ML module: cosine similarity & norms underpin regression, regularization, and search.
"""

_QUIZ = [
    Question(
        "The geometric form of the dot product a·b is…",
        ["|a| + |b|", "|a||b|cos θ", "|a||b|sin θ", "a + b"],
        1,
        "a·b = |a||b|cos θ = Σ aᵢbᵢ. It measures alignment.",
    ),
    Question(
        "If a·b = 0 (and neither is the zero vector), the vectors are…",
        ["parallel", "perpendicular (orthogonal)", "equal", "opposite"],
        1,
        "cos θ = 0 means θ = 90°, so the vectors are orthogonal.",
    ),
    Question(
        "Cosine similarity ignores which property of the vectors?",
        ["their direction", "their magnitudes (length)", "their sign", "their dimension"],
        1,
        "Dividing by the norms removes length — only the direction (angle) matters.",
    ),
    Question(
        "The L2 norm of the vector (3, 4) is…",
        ["7", "5", "12", "1"],
        1,
        "√(3² + 4²) = √25 = 5.",
    ),
    Question(
        "A neuron's pre-activation z = w·x + b uses which operation between w and x?",
        ["element-wise max", "the dot product", "the cross product", "concatenation"],
        1,
        "z is the dot product of weights and inputs, plus the bias.",
    ),
    Question(
        "A layer computing Wx (W has one row per neuron) is doing…",
        ["a single dot product", "many dot products at once (one per row/neuron)",
         "a sort", "an activation"],
        1,
        "Matrix-vector multiply = a batch of dot products, one per neuron — the core deep-learning op.",
    ),
    Question(
        "Which norm encourages sparse solutions (many exact zeros)?",
        ["L2 (Euclidean)", "L1 (Manhattan)", "L∞", "the dot product"],
        1,
        "L1 (lasso) drives weights to exactly zero; L2 (ridge) shrinks but rarely zeroes them.",
    ),
    Question(
        "Compute (1, 2, 3) · (0, 1, 0).",
        ["0", "2", "6", "5"],
        1,
        "1·0 + 2·1 + 3·0 = 2 — the dot product 'selects' the 2nd component.",
    ),
]

VECTORS = Lesson(
    key="vectors_dot",
    title="Vectors & the dot product",
    theory=_THEORY,
    quiz=_QUIZ,
    tasks=_TASKS,
    references=_REFERENCES,
)

REGISTRY = {VECTORS.key: VECTORS}
