# Datasets

Small datasets live here (git-ignored — they're regenerable/large). Tiers 0–1
generate their own toy data in code (logic gates, XOR), so nothing is needed yet.

When you reach them:
- **MNIST** (e12–e14): 28×28 handwritten digits. Fetch via your DL framework's
  dataset utilities, or download the raw IDX files here.
- **Tiny text corpus** (e15, e21): e.g. the Tiny Shakespeare file used by
  nanoGPT/makemore — a single `input.txt` is enough to train a char-level model.
