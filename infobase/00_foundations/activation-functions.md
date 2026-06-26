# Activation Functions

The nonlinearity φ applied to a neuron's pre-activation `z`. Without it, depth is
pointless (linear ∘ linear = linear). The choice affects expressiveness and,
crucially, *trainability* (gradient flow).

| φ | formula | range | derivative | notes |
|---|---------|-------|------------|-------|
| step | 1 if z≥0 else 0 | {0,1} | 0 a.e. | not differentiable → can't backprop |
| sign | ±1 | {−1,1} | 0 a.e. | bipolar perceptron |
| sigmoid | 1/(1+e^−z) | (0,1) | σ(1−σ), max 0.25 | saturates → vanishing gradients |
| tanh | (e^z−e^−z)/(e^z+e^−z) | (−1,1) | 1−tanh² | zero-centered; still saturates |
| ReLU | max(0,z) | [0,∞) | 1 if z>0 else 0 | cheap, no saturation on +side; can "die" |
| GELU | z·Φ(z) | ≈[−0.17,∞) | smooth | default in Transformers/GPT |
| softmax | e^zᵢ/Σe^zⱼ | (0,1), sums to 1 | Jacobian | turns logits into a distribution (output layer) |

## Why saturation hurts (the key insight for e07)
Backprop multiplies local derivatives along the path. sigmoid' ≤ 0.25, so through
L sigmoid layers the gradient is scaled by ≤ 0.25^L → it **vanishes** for deep
nets, and learning in early layers stalls. ReLU's derivative is 1 on the active
side, so gradients pass through undiminished — a big reason deep nets became
trainable.

## Practical defaults
- Hidden layers: **ReLU** (or GELU in Transformers).
- Binary output: **sigmoid** (+ BCE loss).
- Multiclass output: **softmax** (+ cross-entropy loss).

**Links:** experiment [e07](../../experiments/tier1_backprop/e07_activation_zoo/)
· code `core/activations.py` · related: [backpropagation.md](backpropagation.md),
[loss-functions.md](loss-functions.md)
