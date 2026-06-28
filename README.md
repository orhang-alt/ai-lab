# AI Lab

A personal lab for learning **artificial neural networks and LLMs from scratch** —
starting from a single artificial neuron and building up, one experiment at a time,
to a small GPT.

## Design principle: one library, many experiments

There is **one growing library** (`core/`) of reusable building blocks. Every
experiment under `experiments/` only *orchestrates and observes* — it imports from
`core/` and never reimplements a neuron, a layer, an optimizer, etc. A neuron you
write in Tier 0 is literally the same class that, many layers deep, becomes a
Transformer's MLP block.

```
core/            the building blocks (grows each tier)
experiments/     the hierarchy of studies (tier0 → tier5)
infobase/        your knowledge base, written in your own words
papers/          annotated reading list + PDFs
datasets/        small datasets
tests/           gradient checks etc. — proves core/ is correct
```

## Setup

```bash
cd ai-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[viz,dev]"      # numpy + matplotlib + pytest
```

> **Python version note:** Tiers 0–1 (the "from scratch" work) need only NumPy and
> run on any modern Python, including 3.14. For Tiers 3+ (PyTorch) you may need a
> separate Python 3.12 venv if PyTorch wheels lag the newest interpreter. See
> `requirements.txt`.

## Run your first experiment

```bash
python experiments/tier0_neuron/e01_single_neuron/run.py
```

Then read [`experiments/tier0_neuron/e01_single_neuron/README.md`](experiments/tier0_neuron/e01_single_neuron/README.md)
— it's the template for every experiment that follows.

## Or launch the GUI

A local Streamlit dashboard ties it all together — ANN, classical ML, and math
tracks; interactive playgrounds; experiment runner; infobase reader; roadmap
progress; and test runner:

```bash
pip install -e ".[gui]"
./start.sh
```

See [`gui/README.md`](gui/README.md).

## How to use this lab

1. Read the matching `infobase/` note for the concept.
2. Open the experiment's `README.md` — it states a **hypothesis** and **what to observe**.
3. Implement / run `run.py`, record results in the experiment's `notes.md`.
4. When you build something reusable, promote it into `core/` and add a test.
5. Use `learning/templates/checkpoint.md` and `learning/questions/` to prove you
   can reconstruct the idea from memory.

See [`ROADMAP.md`](ROADMAP.md) for the full plan and your progress checklist.
