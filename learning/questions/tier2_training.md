# Tier 2 Questions — Training Dynamics

Answer these aloud before looking at code.

1. Why can a learning rate be too small or too large?
2. What does momentum remember from previous steps?
3. What problem does Adam solve better than plain SGD?
4. Why is all-zero initialization bad for multi-neuron layers?
5. Why does Xavier initialization fit tanh/sigmoid better than ReLU?
6. Why does He initialization fit ReLU-family activations?
7. What is overfitting in terms of train loss and validation loss?
8. How does L2 regularization change the update?
9. What does dropout force a network to stop relying on?
10. When a model fails, how do you tell optimization failure from generalization failure?

---

## Answer key

Try each aloud first, then check.

<details><summary>1. Learning rate too small vs too large</summary>
Too small → training crawls and may not converge in the steps you have. Too large → steps overshoot the minimum, so the loss oscillates or diverges. There's a stable band in between.
</details>

<details><summary>2. What momentum remembers</summary>
A running average (velocity) of past gradients. It keeps moving in a consistent direction and cancels back-and-forth oscillations.
</details>

<details><summary>3. What Adam solves better than plain SGD</summary>
It adapts a per-parameter step size (dividing by a running RMS of each gradient), so it copes with ill-conditioned / ravine losses and with gradients of very different scales — far better than one global learning rate.
</details>

<details><summary>4. Why all-zero init is bad for multi-neuron layers</summary>
Every neuron in the layer then computes the same output and receives the same gradient, so they stay identical forever — the symmetry never breaks. Random init breaks it so units can specialize.
</details>

<details><summary>5. Why Xavier fits tanh/sigmoid</summary>
Xavier keeps the activation variance roughly constant assuming a linear regime symmetric around 0 — exactly where tanh/sigmoid operate near the origin — so it suits them.
</details>

<details><summary>6. Why He fits ReLU-family activations</summary>
ReLU zeros about half its inputs, halving the signal variance. He init scales the weights up by √2 to compensate, keeping the variance stable through ReLU layers.
</details>

<details><summary>7. Overfitting in terms of train vs validation loss</summary>
Train loss keeps falling while validation loss stops falling and starts rising — the gap widens. The model is memorizing the training data instead of generalizing.
</details>

<details><summary>8. How L2 changes the update</summary>
It adds λ‖w‖² to the loss, so the gradient gains a +2λw term and each weight is shrunk toward 0 a little every step ("weight decay").
</details>

<details><summary>9. What dropout stops the network relying on</summary>
Any single unit or co-adapted feature: by randomly zeroing activations it forces the network to spread the representation across many units (an implicit ensemble), which improves generalization.
</details>

<details><summary>10. Optimization failure vs generalization failure</summary>
Check the TRAIN loss first. If it won't go down, it's an optimization failure (can't even fit). If train loss is low but VALIDATION loss is high, it's a generalization failure (fits but doesn't transfer).
</details>

