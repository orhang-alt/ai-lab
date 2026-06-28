# e21 — nanoGPT-style char-level GPT

**Tier 4 → the real thing.** The GUI's *Tiny GPT* page explains a Transformer and fakes the
next-token model with a fixed-window net. This experiment is the **actually-trained**
version: a real GPT in PyTorch, learning by backprop on a GPU/CPU.

## Hypothesis
A small stack of **causal multi-head self-attention** blocks, trained only to **predict the
next character** (cross-entropy), will learn the structure of a corpus well enough to
**generate** new, plausible text — demonstrating that "attention + next-token training" is
all an LLM fundamentally is.

## Method
- **Model:** token + positional embeddings → *N* × [LayerNorm → causal self-attention →
  residual → LayerNorm → MLP → residual] → LayerNorm → linear head over the vocabulary.
  (Scaled dot-product attention `softmax(QKᵀ/√d)` with a triangular causal mask — exactly
  the GUI *Attention* / *Tiny GPT* pages, in PyTorch.)
- **Objective:** next-token **cross-entropy**; **optimizer:** AdamW.
- **Device:** Apple **MPS** / CUDA / CPU (auto-detected).
- **Data:** a built-in offline corpus by default; `--download` fetches tiny-shakespeare (~1 MB).

## Run
```bash
pip install -r requirements-dl.txt          # torch (3.14 ok; else use a 3.12 venv)

python experiments/tier4_attention/e21_nanogpt/run.py                  # quick demo (~15 s on MPS)
python experiments/tier4_attention/e21_nanogpt/run.py --download --iters 3000 --n_layer 4
```

## Observe
Even the 200-iter demo drops the loss fast and already echoes the corpus's grammar:

```
device=mps  vocab=29  model: 3L/4H/128d
  step   0   train 3.480   val 3.498
  step 200   train 0.348   val 0.347
sample: The farmer built the long road yet the queen carried the small house. ...
```

With `--download` + a few thousand iters it produces Shakespeare-flavoured English.

## Conclusion
This is **scaled training**: real attention, real backprop, a real optimizer, on real
hardware. It's the same machinery as a production LLM — only the data, parameter count, and
compute differ. End of the lab's core arc: **neuron → MLP → backprop → attention →
Transformer → a GPT that writes.**

## Next (roadmap)
e22 BPE tokenization · e23 sampling (top-k/top-p) · e24 fine-tuning · e25 LoRA · e27 RAG + KV-cache.
