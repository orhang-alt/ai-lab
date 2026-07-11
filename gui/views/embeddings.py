"""Embeddings & RAG — meaning as geometry (ANN module, roadmap e27).

Text becomes vectors so that **similar meaning → nearby vectors**; similarity is the dot
product / cosine (Math X1). On top of that sits **retrieval-augmented generation (RAG)**:
embed your documents, retrieve the most similar ones to a question, and feed them to the LLM
so it answers grounded in facts. The Live tab runs cosine-similarity retrieval over a small
knowledge base.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import lessons

KB = [
    "Gradient descent trains a neural network by repeatedly stepping downhill on the loss.",
    "A transformer uses self-attention to weigh relationships between all tokens in a sequence.",
    "Backpropagation computes the gradient of the loss for every weight in one backward pass.",
    "Photosynthesis lets plants turn sunlight, water and carbon dioxide into glucose.",
    "The mitochondrion is the powerhouse of the cell, producing energy as ATP.",
    "Mount Everest is the tallest mountain above sea level, on the Nepal-China border.",
    "The Pacific is the largest and deepest of Earth's oceans.",
    "Water boils at 100 degrees Celsius at sea-level atmospheric pressure.",
    "The French Revolution began in 1789 and overthrew the French monarchy.",
    "Python is a high-level programming language known for its readable syntax.",
]


@st.cache_resource(show_spinner=False)
def build_index():
    vec = TfidfVectorizer(stop_words="english")
    M = vec.fit_transform(KB)
    return vec, M


def retrieve(query, k):
    vec, M = build_index()
    sims = cosine_similarity(vec.transform([query]), M)[0]
    order = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in order]


_RAG_SVG = '''<div style="text-align:center;margin:0.5rem 0"><svg viewBox="0 0 720 150" style="width:100%;max-width:720px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="RAG pipeline: the question is embedded, used to search a vector database of embedded documents, the top-k chunks are retrieved and placed in the prompt alongside the question, and the LLM generates a grounded answer."><defs><marker id="rag" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#5B8FC2"/></marker></defs><rect x="1" y="1" width="718" height="148" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g font-family="sans-serif" font-size="11.5" text-anchor="middle"><rect x="18" y="56" width="92" height="40" rx="8" fill="#E6F1FB" stroke="#5B8FC2"/><text x="64" y="80" fill="#0C447C">question</text><rect x="140" y="56" width="96" height="40" rx="8" fill="#FBEAD6" stroke="#9A6A2A"/><text x="188" y="74" fill="#5A3E14">embed →</text><text x="188" y="88" fill="#5A3E14">vector</text><rect x="266" y="44" width="120" height="64" rx="8" fill="#FFFFFF" stroke="#5B8FC2"/><text x="326" y="68" fill="#0C447C">vector DB</text><text x="326" y="84" fill="#6B6A66" font-size="9.5">cosine search</text><text x="326" y="98" fill="#6B6A66" font-size="9.5">(embedded docs)</text><rect x="416" y="56" width="104" height="40" rx="8" fill="#D7EFE5" stroke="#1D9E75"/><text x="468" y="74" fill="#0E5E45">top-k chunks</text><text x="468" y="88" fill="#0E5E45" font-size="9.5">+ question</text><rect x="550" y="56" width="70" height="40" rx="8" fill="#FBEAF0" stroke="#C0507A"/><text x="585" y="80" fill="#8A2351">LLM</text><rect x="648" y="56" width="58" height="40" rx="8" fill="#D7EFE5" stroke="#1D9E75"/><text x="677" y="80" fill="#0E5E45">answer</text></g><g stroke="#5B8FC2" stroke-width="1.7" fill="none"><line x1="110" y1="76" x2="138" y2="76" marker-end="url(#rag)"/><line x1="236" y1="76" x2="264" y2="76" marker-end="url(#rag)"/><line x1="386" y1="76" x2="414" y2="76" marker-end="url(#rag)"/><line x1="520" y1="76" x2="548" y2="76" marker-end="url(#rag)"/><line x1="620" y1="76" x2="646" y2="76" marker-end="url(#rag)"/></g><text x="326" y="128" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#9C9B95">retrieve relevant facts, then let the LLM answer grounded in them</text></svg></div>'''


_THEORY = r"""
## 1. Meaning as geometry

An **embedding** maps a piece of text (a word, sentence, or document) to a dense **vector**,
arranged so that **similar meaning → nearby vectors**. Closeness is measured with the
**cosine / dot product** (Math X1): aligned vectors → high similarity, orthogonal → unrelated.
Good learned embeddings capture *semantics*, not just shared words — `king − man + woman ≈
queen`, and "car" ≈ "automobile" even with no letters in common.

## 2. How embeddings are learned

The trick: train vectors so that text appearing in **similar contexts** ends up close.
- **word2vec / GloVe** — predict a word from its neighbours (skip-gram); the hidden vectors
  become the embeddings.
- **Sentence/document embedders** — **contrastive** learning: pull paraphrases together,
  push unrelated pairs apart.
- An LLM's **token embeddings** (Attention page) are a learned special case.

## 3. Cosine similarity & vector search

To find relevant text, embed everything and rank by cosine to the query — "**semantic
search**" is just **nearest-neighbours in embedding space**. Over millions of vectors,
approximate-nearest-neighbour indexes (FAISS, HNSW) make it fast; that store is a **vector
database**.

## 4. RAG — retrieval-augmented generation

LLMs **hallucinate** and have a frozen **knowledge cutoff**. RAG fixes both without
retraining:

<RAG/>

1. **Index** — split your documents into chunks, embed them, store in a vector DB.
2. **Retrieve** — embed the question, fetch the top-k most similar chunks.
3. **Augment** — put those chunks into the prompt next to the question.
4. **Generate** — the LLM answers **grounded** in the retrieved facts (and can cite them).

Update knowledge by updating the database — no fine-tuning required.

## 5. Chunking & practicalities

Quality hinges on the unglamorous parts: how you **chunk** documents (too big wastes context,
too small loses meaning), the **embedder** quality, how many chunks to retrieve (the **context
budget** is in tokens — Tokenization page), and optional **re-ranking** of candidates.

## 6. RAG vs. fine-tuning

Use **RAG** to supply **facts** (fresh, proprietary, citable). Use **fine-tuning** to teach
**behaviour/style** (the post-training page). They're complementary — many systems do both.

## 7. About this demo

The Live tab uses **lexical TF-IDF** vectors (self-contained, no model download), so it
matches on shared words rather than deep meaning. The **pipeline is exactly RAG** — embed →
cosine-rank → retrieve → would-be prompt; real systems simply swap in **learned semantic**
embeddings. *(Roadmap e27; ties to Math X1 and the Attention page.)*
"""

_QUIZ = [
    lessons.Question(
        "What is an embedding?",
        ["a compressed image", "a vector for a piece of text, placed so similar meanings are nearby",
         "a type of loss function", "a tokenizer"], 1,
        "Embeddings turn text into vectors where geometric closeness (cosine) ≈ semantic similarity (Math X1)."),
    lessons.Question(
        "Semantic search ranks documents by…",
        ["alphabetical order", "cosine similarity (nearest neighbours) between the query and document embeddings",
         "file size", "publication date"], 1,
        "Embed query + docs, then rank by cosine/dot product — nearest neighbours in embedding space."),
    lessons.Question(
        "RAG reduces hallucination and stale knowledge by…",
        ["making the model bigger", "retrieving relevant documents and putting them in the prompt so the LLM answers grounded in them",
         "lowering the temperature", "removing the tokenizer"], 1,
        "Retrieve top-k chunks → augment the prompt → the LLM answers from supplied facts; update the DB, not the weights."),
    lessons.Question(
        "RAG vs. fine-tuning, roughly:",
        ["they're identical", "RAG supplies facts/knowledge; fine-tuning teaches behaviour/style",
         "fine-tuning is always better", "RAG retrains the model"], 1,
        "RAG injects knowledge at query time; fine-tuning changes the weights to shape behaviour."),
]

_TASKS = r"""
### In the Retrieve tab
1. Ask **"how do neural networks learn?"** — which facts come back on top, and do their
   cosine scores make sense?
2. Ask something the knowledge base **doesn't** contain (e.g. "who painted the Mona Lisa?").
   Note the low scores — a real RAG system would say "not in my sources" rather than guess.
3. Because this demo is **lexical** (TF-IDF), try a query that means the same thing but uses
   **different words** than the fact — watch it miss. That gap is exactly what *learned*
   semantic embeddings close.

### Concept
4. Sketch the four RAG steps (index → retrieve → augment → generate).
5. When would you reach for RAG, and when for fine-tuning?
"""

_REFS = r"""
- Mikolov et al. (2013) — *word2vec*; Pennington et al. (2014) — *GloVe*.
- Reimers & Gurevych (2019) — *Sentence-BERT* (sentence embeddings).
- Lewis et al. (2020) — *Retrieval-Augmented Generation*.
- FAISS / HNSW (vector search); Hugging Face *sentence-transformers*.
- In this lab: Math **X1** (dot product / cosine), **Attention** (token embeddings),
  **Tokenization**, and the post-training page (fine-tuning vs RAG).
"""


st.title("Embeddings & RAG — meaning as geometry")
st.caption("Text → vectors where similar meaning sits nearby (cosine, Math X1); retrieval-"
           "augmented generation feeds the most similar facts to the LLM. Live search below.")

lessons.predict(
    'You ask something the knowledge base does **not** contain. What will the top **cosine similarity** look like — and what should a good RAG system do?',
    "The top similarity will be **low** — nothing is close. A good RAG system notices that and says *'I don't know'* instead of grounding an answer in irrelevant text (which is how hallucinations sneak in).",
)

tab_live, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🔎 Retrieve", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_live:
    st.markdown("A tiny **knowledge base** of facts. Ask a question — it's embedded and "
                "matched by **cosine similarity**, and the top hits become the LLM's context.")
    with st.expander("the knowledge base (10 facts)"):
        st.markdown("\n".join(f"- {d}" for d in KB))

    cc = st.columns([0.75, 0.25])
    query = cc[0].text_input("question", value="how do neural networks learn?", key="emb_q")
    k = cc[1].slider("top-k", 1, 5, 3, key="emb_k")

    hits = retrieve(query, k)
    st.caption("Retrieved facts (ranked by cosine similarity):")
    st.dataframe(
        pd.DataFrame([{"similarity": f"{s:.2f}", "retrieved passage": KB[i]} for i, s in hits]),
        hide_index=True, width="stretch")

    best_sim = hits[0][1] if hits else 0.0
    context = "\n".join(f"- {KB[i]}" for i, _ in hits)
    st.caption("The prompt RAG would send to the LLM:")
    st.code(f"Answer the question using ONLY the context.\n\nContext:\n{context}\n\n"
            f"Question: {query}\nAnswer:", language=None)

    if best_sim < 0.08:
        st.warning("Top similarity is very low — the answer probably isn't in the knowledge "
                   "base. A good RAG system says so instead of hallucinating.",
                   icon=":material/help:")
    else:
        st.success(f"Best match scored {best_sim:.2f}. The LLM now answers grounded in these "
                   "retrieved facts — no retraining, and updatable by editing the KB.",
                   icon=":material/check_circle:")
    st.info("This demo uses lexical TF-IDF vectors, so it matches shared *words*. The "
            "pipeline is exactly RAG; real systems swap in **learned semantic** embeddings "
            "that also match paraphrases and synonyms.", icon=":material/database:")

with tab_theory:
    st.markdown(_THEORY.replace("<RAG/>", _RAG_SVG), unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="embeddings")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
