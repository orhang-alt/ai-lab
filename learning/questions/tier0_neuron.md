# Tier 0 Questions — The Neuron

Answer these aloud before looking at code.

1. What does a neuron compute before the activation function?
2. Why is the dot product a measure of alignment between input and weights?
3. What role does the bias play geometrically?
4. Why can one neuron represent AND and OR but not XOR?
5. What does the perceptron update do when the prediction is too low?
6. Why does the perceptron converge for linearly separable data?
7. What does "linearly separable" mean in two dimensions?
8. How would you move the decision boundary without changing its slope?
9. What information is lost when you use a step activation?
10. Why is XOR the first real reason to need layers?

---

## Answer key

Try each aloud first, then check.

<details><summary>1. Before the activation</summary>
The pre-activation z = w·x + b — a weighted sum of the inputs plus the bias (an affine function).
</details>

<details><summary>2. Dot product = alignment</summary>
w·x = ‖w‖‖x‖cosθ: largest when x points along w, zero when perpendicular, negative when opposed — so it scores how much the input aligns with the weight direction.
</details>

<details><summary>3. The bias, geometrically</summary>
It shifts the decision boundary (the line w·x + b = 0) toward or away from the origin without changing its orientation. The offset is −b/‖w‖.
</details>

<details><summary>4. AND/OR yes, XOR no</summary>
AND and OR are linearly separable — one straight line puts the 1s on one side. XOR's two 1s sit on opposite corners, so no single line separates them, and one neuron draws only one line.
</details>

<details><summary>5. Perceptron update when the prediction is too low</summary>
With y − pred > 0 it adds the input to the weights (w += lr·(y−pred)·x) and raises the bias, nudging the boundary so that input scores higher next time.
</details>

<details><summary>6. Why it converges for separable data</summary>
Every mistake moves the boundary toward a separating position; the perceptron convergence theorem guarantees this stops after finitely many updates whenever a separator exists.
</details>

<details><summary>7. "Linearly separable" in 2D</summary>
The two classes can be split by a single straight line — all of one class on one side, all of the other on the other side.
</details>

<details><summary>8. Move the boundary without changing its slope</summary>
Change only the bias b (it translates the line). Changing the weights rotates it.
</details>

<details><summary>9. What a step activation throws away</summary>
The magnitude / confidence of z: step emits only 0 or 1, discarding how far the point is from the boundary. It also has zero gradient, so it can't be trained by gradient descent.
</details>

<details><summary>10. Why XOR forces layers</summary>
No single neuron (one line) can separate XOR. A hidden layer invents a new feature (e.g. AND) that makes the problem linearly separable for the output neuron — the first real need for depth.
</details>

