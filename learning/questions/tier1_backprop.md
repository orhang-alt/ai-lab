# Tier 1 Questions — Backprop and Autograd

Answer these aloud before looking at code.

1. Why does `backward()` start by setting the output gradient to `1.0`?
2. Why does a computation graph need a topological order?
3. Why do gradients accumulate with `+=` instead of assignment?
4. For `d = a*b + c`, why are `dd/da = b`, `dd/db = a`, and `dd/dc = 1`?
5. What does a local derivative rule know, and what does it not know?
6. Why does ReLU pass gradient only on the active side?
7. Why can an MLP solve XOR?
8. What exactly does `zero_grad()` prevent?
9. Why does cross-entropy learn faster than MSE for confident wrong classification?
10. What does gradient checking prove, and what does it not prove?

---

## Answer key

Try each aloud first, then check.

<details><summary>1. Why backward() seeds the output gradient with 1.0</summary>
It seeds the chain rule: the derivative of the output with respect to itself is 1 (dL/dL = 1). Every other gradient is that seed multiplied backward through local derivatives.
</details>

<details><summary>2. Why a topological order is needed</summary>
A node's gradient can only be finalized once every node that uses it downstream has contributed. Reverse-topological order guarantees each consumer is processed before the node, so its accumulated gradient is complete.
</details>

<details><summary>3. Why gradients accumulate with +=</summary>
A value feeding several operations receives a gradient from each path; the multivariable chain rule sums those contributions, so backward() adds (+=) rather than overwriting.
</details>

<details><summary>4. Derivatives of d = a*b + c</summary>
Treating the others as constant: ∂d/∂a = b, ∂d/∂b = a (product rule), and ∂d/∂c = 1 (sum rule).
</details>

<details><summary>5. What a local rule does and doesn't know</summary>
It knows only its own operation and inputs — its local derivative. It does not know the overall loss or its place in the graph; the incoming upstream gradient supplies that. Local derivative × upstream gradient = the chain rule.
</details>

<details><summary>6. Why ReLU passes gradient only on the active side</summary>
ReLU is the identity for positive inputs (slope 1) and flat for negative ones (slope 0), so it passes the gradient unchanged where it is active and blocks it where it is off.
</details>

<details><summary>7. Why an MLP can solve XOR</summary>
Its hidden layer builds new intermediate features (nonlinear combinations of the inputs) that make XOR linearly separable for the output neuron — something one straight line cannot do.
</details>

<details><summary>8. What zero_grad() prevents</summary>
It clears the .grad accumulated on the previous step. Because gradients accumulate with +=, skipping it would add this step's gradients onto the last step's and corrupt the update.
</details>

<details><summary>9. Why cross-entropy learns faster than MSE when confidently wrong</summary>
With sigmoid + MSE a confidently wrong output sits in the flat tail where the derivative ≈ 0, so the gradient nearly vanishes and learning stalls. Cross-entropy's gradient is (p − y), which is largest exactly when the model is confident and wrong — so it corrects quickly.
</details>

<details><summary>10. What gradient checking does and doesn't prove</summary>
It proves your backprop implementation agrees with the numerical (finite-difference) derivative — i.e. the code is correct. It does not prove the model or the math is the right choice, and it won't catch a bug shared by both the analytic and numerical paths.
</details>

