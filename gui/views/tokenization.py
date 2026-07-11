"""Tokenization — how text becomes tokens (ANN module, roadmap e22).

Before any LLM math, text must be cut into **tokens**. This page compares character, word,
and **sub-word (BPE)** tokenization, and trains a tiny Byte-Pair-Encoding tokenizer live:
start from characters and repeatedly merge the most frequent adjacent pair, watching common
strings fuse into single tokens.
"""

import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import pandas as pd
import streamlit as st

import lessons

CORPUS = (
    "the learner trains the model and the model learns the pattern. "
    "training the network means tuning the weights so the loss is lower. "
    "the teacher tested the learner and the learner tested the model. "
    "lower the learning rate when the training loss is jumping around. "
    "the smaller model learned faster but the larger model generalized better. "
    "tokenizing the text lets the transformer read the words as numbers."
) * 4
END = "·"   # visible end-of-word marker


def _stats(words):
    pairs = Counter()
    for sym, c in words.items():
        for i in range(len(sym) - 1):
            pairs[(sym[i], sym[i + 1])] += c
    return pairs


def _merge_sym(sym, pair):
    out, i = [], 0
    while i < len(sym):
        if i < len(sym) - 1 and (sym[i], sym[i + 1]) == pair:
            out.append(sym[i] + sym[i + 1]); i += 2
        else:
            out.append(sym[i]); i += 1
    return tuple(out)


@st.cache_data(show_spinner=False)
def train_bpe(n_merges):
    wc = Counter(CORPUS.split())
    words = {tuple(list(w) + [END]): c for w, c in wc.items()}
    merges = []
    for _ in range(n_merges):
        pairs = _stats(words)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        words = {_merge_sym(s, best): c for s, c in words.items()}
        merges.append(best)
    return merges


def tokenize_word(word, merges):
    sym = tuple(list(word) + [END])
    for pair in merges:
        sym = _merge_sym(sym, pair)
    return sym


def bpe_tokens(text, merges):
    toks = []
    for w in text.split():
        toks.extend(tokenize_word(w, merges))
    return toks


_THEORY = r"""
## 1. Why tokenize at all

A neural network only eats **numbers**, so the first step of any language model is to cut
raw text into a sequence of units — **tokens** — that get mapped to integer IDs and then to
**embedding** vectors (Attention page). *How* you cut the text is a real design choice with
big consequences for vocabulary size, sequence length, and what the model can spell or
generalize.

## 2. Characters

Split into individual characters. **Pros:** tiny vocabulary (~100), and **no
out-of-vocabulary** problem — any string is representable. **Cons:** sequences get **very
long**, and attention costs $O(n^2)$ in length (Attention §9); the model also has to learn
spelling and word structure from scratch. *(This is exactly what the lab's Tiny GPT uses —
simple, but inefficient at scale.)*

## 3. Words

Split on whitespace. **Pros:** short sequences, units carry meaning. **Cons:** a **huge**
vocabulary (100k+), it chokes on **unseen words** (OOV → a useless `<unk>`), and it wastes
capacity on morphology — *run, runs, running, runner* all become unrelated entries.

## 4. Sub-word (BPE) — the sweet spot

Modern LLMs use **sub-word** tokens, most often via **Byte-Pair Encoding (BPE)**. Idea: start
from characters and **repeatedly merge the most frequent adjacent pair** into a new token.
Frequent whole words (`the`, `ing`, `tion`) end up as single tokens; rare words break into a
few pieces; **nothing is ever OOV** (worst case, it falls back to characters/bytes). You get
a fixed, tunable vocabulary (~30k–100k) that balances sequence length against vocab size.
(GPT uses byte-level BPE; relatives are WordPiece, Unigram, SentencePiece.)

## 5. The BPE algorithm

**Train** (learn the merges):
1. Represent each word as its sequence of characters (+ an end-of-word marker).
2. Count every **adjacent pair** across the corpus.
3. **Merge** the most frequent pair everywhere into one new symbol; record the merge.
4. Repeat for a set number of merges → an ordered **merge list** + the vocabulary.

**Encode** new text: apply the learned merges **in the same order**, greedily, to each word.
The slider in the demo *is* step 4's count — slide it up and watch characters fuse into
sub-words and then whole words.

## 6. Special tokens

Tokenizers reserve a few non-text tokens: **`<bos>` / `<eos>`** (begin/end of sequence),
**`<pad>`** (fill a batch to equal length), **`<unk>`** (unknown — rare with BPE), and chat
**role markers** (system/user/assistant) that structure a conversation for an instruct model.

## 7. Why it matters in practice

- **Context windows and API bills are counted in tokens**, not characters or words
  (English averages ~**4 characters per token**).
- **Other languages and code tokenize less efficiently** — more tokens for the same meaning,
  so higher cost and less effective context.
- Many famous LLM quirks (miscounting letters in a word, fumbling arithmetic) trace back to
  the fact that the model sees **tokens, not characters**.

*(Roadmap e22. Next: this feeds the Attention → Tiny GPT → Decoding pages.)*
"""

_QUIZ = [
    lessons.Question(
        "Why do language models tokenize text first?",
        ["to compress the file", "to turn text into integer IDs (then embedding vectors) the network can process",
         "to fix spelling", "to remove punctuation"], 1,
        "Networks need numbers; tokenize → IDs → embeddings is the entry point (Attention page)."),
    lessons.Question(
        "The main drawback of *word-level* tokenization is…",
        ["sequences are too long", "a huge vocabulary and no way to handle unseen (OOV) words",
         "it has no vocabulary", "it can't represent 'the'"], 1,
        "Word vocab balloons and unknown words become <unk>; sub-word fixes both."),
    lessons.Question(
        "Byte-Pair Encoding builds its vocabulary by…",
        ["random sampling", "starting from characters and repeatedly merging the most frequent adjacent pair",
         "keeping only the top 100 words", "splitting on spaces"], 1,
        "Greedy frequency merges grow common strings into single tokens; rare words stay in pieces."),
    lessons.Question(
        "Sub-word tokenization avoids out-of-vocabulary words because…",
        ["it has every word", "any unseen word can always fall back to smaller pieces / characters",
         "it ignores unknown words", "it lowercases everything"], 1,
        "Worst case a word decomposes into characters, so nothing is unrepresentable."),
]

_TASKS = r"""
### In the Tokenize tab
1. Set **merges = 0** — confirm the text is pure **characters**. Slide up and watch the
   first merges: which pairs win first, and why? (Hint: the most *frequent* adjacent pairs.)
2. Find the merge count where common words like `the` become a **single** token. What
   happens to the **token count** of your text as merges increase?
3. Type a **word not in the corpus** (e.g. *quixotic*) — how does BPE tokenize it, and why
   doesn't it become `<unk>`?

### Concept
4. Give one advantage and one disadvantage each of character, word, and sub-word tokenization.
5. Why are an LLM's context length and price measured in *tokens* rather than words?
"""

_REFS = r"""
- Sennrich et al. (2016) — *Neural Machine Translation of Rare Words with Subword Units* (**BPE**).
- Karpathy — *Let's build the GPT Tokenizer* / `minbpe` (BPE from scratch).
- Hugging Face — *Tokenizers* course (BPE / WordPiece / Unigram / SentencePiece).
- OpenAI — `tiktoken` (the byte-level BPE used by GPT models).
- In this lab: **Attention** (tokens → embeddings), **Tiny GPT**, **Decoding**.
"""


st.title("Tokenization — how text becomes tokens")
st.caption("Before any LLM math, text is cut into tokens. Compare character / word / "
           "sub-word, and train a tiny BPE tokenizer live by merging frequent pairs.")

lessons.predict(
    'As you increase the number of **BPE merges**, does your text become *more* tokens or *fewer* — and what grows in return?',
    "**Fewer, longer** tokens (a shorter sequence) — but the **vocabulary** grows. Real LLMs pick a fixed vocab (~30k–100k), the balance point. Unknown words fall back to smaller pieces, so there's no out-of-vocabulary problem.",
)

tab_tok, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["✂️ Tokenize", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_tok:
    st.markdown("Type some text, then slide the number of **BPE merges**: at 0 it's pure "
                "characters; as merges grow, frequent pairs fuse into sub-words and whole "
                f"words. (`{END}` marks a word end.)")
    text = st.text_input("text to tokenize", value="the learner is tokenizing words", key="tk_txt")
    n_merges = st.slider("BPE merges (vocabulary granularity)", 0, 80, 30, key="tk_merges")

    merges = train_bpe(n_merges)
    toks = bpe_tokens(text, merges)
    n_chars = len(text.replace(" ", ""))
    n_words = len(text.split())

    m = st.columns(4)
    m[0].metric("characters", n_chars)
    m[1].metric("words", n_words)
    m[2].metric("BPE tokens", len(toks))
    m[3].metric("merges learned", len(merges))

    st.caption("Your text as BPE tokens:")
    chips = "  ".join(
        "&nbsp;".join(f"<code style='background:#E6F1FB;padding:1px 4px;border-radius:4px'>"
                      f"{t}</code>" for t in tokenize_word(w, merges))
        for w in text.split()
    )
    st.markdown(f"<div style='line-height:2.2'>{chips}</div>", unsafe_allow_html=True)

    if merges:
        st.caption("First learned merges (most frequent pairs, in order):")
        rows = [{"#": i + 1, "merge": f"{a} + {b}", "→ token": a + b}
                for i, (a, b) in enumerate(merges[:14])]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="content")

    st.info("Same text, different granularity: more merges → fewer, longer tokens (shorter "
            "sequence) but a bigger vocabulary. Real LLMs pick a fixed vocab (~30k–100k) — "
            "the balance point. Unknown words just fall back to smaller pieces, so no OOV.",
            icon=":material/content_cut:")

with tab_theory:
    st.markdown(_THEORY, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="tokenization")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** merges = 0 → pure characters. The **most frequent adjacent pairs** win first (e.g. `t`+`h` → `th`, then common endings) — BPE greedily merges the commonest pair each round.

**2.** After enough merges a common word like `the` collapses to a **single** token; as merges rise, your text's **token count drops** (fewer, longer tokens).

**3.** An out-of-corpus word falls back to the **sub-word pieces / characters** BPE already knows, so it's always representable — there's no `<unk>`. That's BPE's headline advantage.""",
        label="Tokenize tab 1–3",
    )
    lessons.solution(
        r"""**4.** **Character**: tiny vocab, zero OOV, but long sequences and little meaning per token. **Word**: meaningful and short, but a huge vocab and an OOV problem. **Sub-word (BPE)**: the balance — modest fixed vocab, no OOV, reasonable sequence length.

**5.** The model consumes **tokens**, not words: compute and context window are per-token, and one word may be several tokens (or a token may span parts of words). Tokens are the true unit of cost, so length and price are measured in them.""",
        label="Concept 4–5",
    )

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
