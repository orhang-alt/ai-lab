# Loss Functions

The scalar the network minimizes — it defines *what "good" means*. Training =
adjust weights to reduce the loss via gradient descent.

## Mean Squared Error (regression)
```
MSE = (1/N) Σ (ŷ − y)²
```
Gradient w.r.t. ŷ is `2(ŷ − y)/N`. Penalizes large errors quadratically.

## Binary Cross-Entropy (binary classification)
With a sigmoid output `p = σ(z)` and target `y ∈ {0,1}`:
```
BCE = −[ y·log(p) + (1−y)·log(1−p) ]
```
The magic: `d(BCE)/dz = p − y`. The awkward `σ'(z)` term **cancels**, so a
confidently-wrong neuron still gets a large gradient. With MSE+sigmoid the
gradient carries a `σ'(z)` factor that → 0 at saturation, so a confidently-wrong
neuron barely learns. This is why classification uses cross-entropy. (You'll
measure this directly in e08.)

## Categorical Cross-Entropy (multiclass)
With softmax over logits and one-hot target:
```
CE = −Σ yᵢ log(softmax(z)ᵢ)
```
And again the gradient simplifies beautifully: `dCE/dz = softmax(z) − onehot(y)`.

## Picking a loss
| task | output activation | loss |
|------|-------------------|------|
| regression | linear | MSE |
| binary class | sigmoid | BCE |
| multiclass | softmax | cross-entropy |

**Links:** experiment [e08](../../experiments/tier1_backprop/e08_loss_functions/)
· code `core/losses.py` · related: [activation-functions.md](activation-functions.md),
[gradient-descent.md](gradient-descent.md)
