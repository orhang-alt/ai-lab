# Papers — Annotated Reading List

Read *just-in-time*: the paper for a concept when you reach its tier. Drop PDFs
in this folder (they're git-ignored; this list is tracked). Don't try to read
everything up front — each entry says *when* and *why*.

## Tier 0 — the neuron
- **McCulloch & Pitts (1943)** — *A Logical Calculus of the Ideas Immanent in
  Nervous Activity.* The first artificial neuron; neurons as logic gates.
- **Rosenblatt (1958)** — *The Perceptron.* The first learning rule. → e02
- **Minsky & Papert (1969)** — *Perceptrons.* Proves the XOR limitation. → e03

## Tier 1 — backprop
- **Rumelhart, Hinton & Williams (1986)** — *Learning representations by
  back-propagating errors.* The algorithm everything rests on. → e04, e06

## Tier 2 — training
- **Glorot & Bengio (2010)** — Xavier initialization. → e10
- **He et al. (2015)** — *Delving Deep into Rectifiers* (He init, PReLU). → e10
- **Srivastava et al. (2014)** — *Dropout.* → e11
- **Kingma & Ba (2014)** — *Adam.* → e09
- **Ioffe & Szegedy (2015)** — *Batch Normalization.*

## Tier 3 — architectures
- **LeCun et al. (1998)** — *Gradient-Based Learning Applied to Document
  Recognition* (LeNet, MNIST). → e14
- **Hochreiter & Schmidhuber (1997)** — *Long Short-Term Memory.* → e15
- **Mikolov et al. (2013)** — *word2vec* (Efficient Estimation of Word
  Representations). → e16
- **He et al. (2015)** — *Deep Residual Learning* (ResNet); residual connections.

## Tier 4 — attention & transformers
- **Bahdanau et al. (2014)** — *Neural MT by Jointly Learning to Align and
  Translate.* First attention. → e17/e18
- **Vaswani et al. (2017)** — ⭐ *Attention Is All You Need.* The Transformer. → e19–e21

## Tier 5 — LLMs
- **Radford et al. (2018)** — GPT. **Radford et al. (2019)** — GPT-2. → e21
- **Devlin et al. (2018)** — *BERT.*
- **Brown et al. (2020)** — *GPT-3* (Language Models are Few-Shot Learners). → e23
- **Hu et al. (2021)** — *LoRA.* → e25
- **Ouyang et al. (2022)** — *InstructGPT* (RLHF). → e26
- **Rafailov et al. (2023)** — *DPO* (Direct Preference Optimization). → e26
- **Lewis et al. (2020)** — *RAG* (Retrieval-Augmented Generation). → e27

## Courses & repos (companions, not papers)
- **Karpathy — "Neural Networks: Zero to Hero"** (videos) + `micrograd`,
  `makemore`, `nanoGPT`. Mirrors this lab's scratch→GPT arc almost exactly.
- **d2l.ai** — *Dive into Deep Learning* (free, math + runnable code).
- **3Blue1Brown** — neural-net & Transformer visual series (intuition).
