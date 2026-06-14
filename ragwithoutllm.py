"""
rag_without_llm.py
==================
A RAG pipeline WITHOUT any LLM — no API key, no internet model call.

Instead of generating a sentence, it RETRIEVES the best matching
chunks and uses "extractive" techniques to surface the answer:

  • BM25 keyword scoring  (sparse retrieval — exact word match)
  • Cosine similarity     (dense retrieval — semantic meaning)
  • Hybrid fusion        (combine both scores)
  • Sentence highlighting (find the single best sentence inside top chunk)

This answers your doubt:
    "Why do we even need an LLM?"

→ Without an LLM, you get the REAL passage from the document.
  There is no hallucination risk, but the output is a raw excerpt —
  not a fluent, synthesised sentence.

The LLM's role is:
  1. Synthesise multiple chunks into a readable answer
  2. Rephrase / paraphrase for clarity
  3. Handle questions that need reasoning across chunks
  4. Answer in a conversational tone

SETUP:
    pip install requests beautifulsoup4 sentence-transformers faiss-cpu

(No API key needed at all.)
"""

import re
import math
import requests
import numpy as np
from collections import Counter
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DOCS_URL      = "https://docs.python.org/3/tutorial/index.html"
CHUNK_SIZE    = 500    # characters per chunk
CHUNK_OVERLAP = 100    # overlap between chunks
TOP_K         = 4      # chunks to retrieve
TOP_SENTENCES = 3      # best sentences to extract from top chunk

QUERY = "What is Python and how is it useful?"


# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DOCUMENT (same as rag_with_gemini.py)
# ─────────────────────────────────────────────────────────────────
def step1_load_documents(base_url: str) -> str:
    print("\n" + "="*60)
    print("STEP 1: LOADING DOCUMENTS")
    print("="*60)
    print(f"  → Fetching: {base_url}")

    headers = {"User-Agent": "Mozilla/5.0 (educational RAG demo)"}
    resp = requests.get(base_url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    base = "https://docs.python.org/3/tutorial/"
    sub_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".html") and not href.startswith("http"):
            sub_links.append(base + href)

    sub_links = list(dict.fromkeys(sub_links))[:6]
    all_text = ""

    for url in [base_url] + sub_links:
        print(f"  → Scraping: {url}")
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            page_soup = BeautifulSoup(r.text, "html.parser")
            for tag in page_soup(["nav", "header", "footer", "script", "style", "aside"]):
                tag.decompose()
            main = (page_soup.find("div", role="main")
                    or page_soup.find("div", class_="body")
                    or page_soup.find("body"))
            text = main.get_text(separator=" ", strip=True) if main else ""
            text = re.sub(r"\s+", " ", text).strip()
            all_text += "\n\n" + text
        except Exception as e:
            print(f"    ⚠ {url}: {e}")

    print(f"  ✓ Total text: {len(all_text):,} characters")
    return all_text.strip()


# ─────────────────────────────────────────────────────────────────
# STEP 2 — CHUNK TEXT
# ─────────────────────────────────────────────────────────────────
def step2_chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    print("\n" + "="*60)
    print("STEP 2: CHUNKING TEXT")
    print("="*60)

    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap

    print(f"  ✓ Chunks created: {len(chunks)}")
    return chunks


# ─────────────────────────────────────────────────────────────────
# STEP 3A — BM25 SPARSE RETRIEVAL
# BM25 is the classic keyword scoring algorithm used by search
# engines (Elasticsearch, Lucene).
#
# Score formula (simplified):
#   Σ  IDF(term) × (tf × (k+1)) / (tf + k×(1 - b + b×docLen/avgDocLen))
#
# where:
#   tf  = how often the term appears in the chunk
#   IDF = how rare the term is across all chunks (rarer = more important)
#   k   = term frequency saturation (default 1.5)
#   b   = document length normalisation (default 0.75)
#
# No embedding model needed. Works great for exact keyword matches.
# ─────────────────────────────────────────────────────────────────
def build_bm25_index(chunks: list[str]) -> dict:
    """Pre-compute BM25 values for the corpus."""
    print("\n  → Building BM25 index (sparse keyword retrieval)...")

    k, b = 1.5, 0.75
    tokenized = [re.findall(r"\b\w+\b", c.lower()) for c in chunks]
    doc_lens  = [len(t) for t in tokenized]
    avg_len   = sum(doc_lens) / len(doc_lens)
    N         = len(chunks)

    # df[term] = number of docs containing that term
    df = Counter()
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1

    return {
        "tokenized": tokenized,
        "doc_lens":  doc_lens,
        "avg_len":   avg_len,
        "df":        df,
        "N":         N,
        "k":         k,
        "b":         b,
    }


def bm25_score(query: str, bm25: dict) -> np.ndarray:
    """Return a BM25 score for every chunk against the query."""
    query_terms = re.findall(r"\b\w+\b", query.lower())
    scores = np.zeros(len(bm25["tokenized"]))

    for term in query_terms:
        if term not in bm25["df"]:
            continue
        # IDF — terms that appear in fewer docs score higher
        idf = math.log((bm25["N"] - bm25["df"][term] + 0.5)
                       / (bm25["df"][term] + 0.5) + 1)

        for i, tokens in enumerate(bm25["tokenized"]):
            tf = tokens.count(term)
            if tf == 0:
                continue
            norm = (bm25["k"] * (1 - bm25["b"]
                    + bm25["b"] * bm25["doc_lens"][i] / bm25["avg_len"]))
            scores[i] += idf * (tf * (bm25["k"] + 1)) / (tf + norm)

    return scores


# ─────────────────────────────────────────────────────────────────
# STEP 3B — DENSE RETRIEVAL (sentence-transformers + FAISS)
# Same as the LLM version — but we stop here and DO NOT call an LLM.
# ─────────────────────────────────────────────────────────────────
def step3_embed_and_index(chunks: list[str]):
    print("\n" + "="*60)
    print("STEP 3: EMBEDDING (dense retrieval) + BM25 (sparse retrieval)")
    print("="*60)

    from sentence_transformers import SentenceTransformer
    import faiss

    print("  → Loading local embedding model: all-MiniLM-L6-v2")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"  → Embedding {len(chunks)} chunks...")
    embeddings = embed_model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # FAISS index for dense retrieval
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  ✓ FAISS index built: {index.ntotal} vectors ({dim}D)")

    # BM25 index for sparse retrieval
    bm25 = build_bm25_index(chunks)
    print(f"  ✓ BM25 index built: vocab size {len(bm25['df'])} terms")

    return embed_model, index, embeddings, bm25


# ─────────────────────────────────────────────────────────────────
# STEP 4 — HYBRID RETRIEVAL (dense + sparse fusion)
# We normalise both score arrays to [0, 1] and add them.
# This is a simplified version of Reciprocal Rank Fusion (RRF).
#
# Why hybrid?
#   Dense  → finds paraphrases ("vehicle" matches "car")
#   Sparse → finds exact keywords ("Python 3.11 release")
#   Hybrid → best of both
# ─────────────────────────────────────────────────────────────────
def step4_hybrid_retrieve(query, embed_model, index, bm25, chunks, top_k):
    print("\n" + "="*60)
    print("STEP 4: HYBRID RETRIEVAL (dense + sparse)")
    print("="*60)
    print(f"  → Query : \"{query}\"")

    # Dense scores (cosine similarity via normalised dot product)
    q_vec = embed_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    dense_scores, _ = index.search(q_vec, len(chunks))
    dense_arr = np.zeros(len(chunks))
    for i, score in zip(_[0], dense_scores[0]):
        dense_arr[i] = score

    # Sparse scores (BM25 keyword match)
    sparse_arr = bm25_score(query, bm25)

    # Normalise both to [0, 1] so they're on the same scale
    def normalise(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    dense_norm  = normalise(dense_arr)
    sparse_norm = normalise(sparse_arr)

    # Hybrid = 0.6 × dense + 0.4 × sparse
    # (weight dense higher since semantic matching is more robust)
    hybrid = 0.6 * dense_norm + 0.4 * sparse_norm

    top_indices = np.argsort(hybrid)[::-1][:top_k]

    print(f"\n  ✓ Top {top_k} chunks (hybrid ranked):")
    retrieved = []
    for rank, idx in enumerate(top_indices, start=1):
        print(f"\n  [{rank}] Chunk #{idx}")
        print(f"       Dense  score : {dense_arr[idx]:.4f}")
        print(f"       Sparse score : {sparse_arr[idx]:.4f}")
        print(f"       Hybrid score : {hybrid[idx]:.4f}")
        print(f"       Preview      : \"{chunks[idx][:120]}...\"")
        retrieved.append((chunks[idx], hybrid[idx]))

    return retrieved


# ─────────────────────────────────────────────────────────────────
# STEP 5 — EXTRACTIVE ANSWER (no LLM needed)
# Instead of generating text, we:
#   1. Take the top retrieved chunks
#   2. Split them into sentences
#   3. Score each sentence against the query (cosine similarity)
#   4. Return the top N most relevant sentences as the "answer"
#
# This is what search engines like Google used to do (featured
# snippets are essentially extractive RAG).
# ─────────────────────────────────────────────────────────────────
def step5_extractive_answer(query, retrieved_chunks_with_scores, embed_model, top_n):
    print("\n" + "="*60)
    print("STEP 5: EXTRACTIVE ANSWER (no LLM — sentence scoring)")
    print("="*60)

    # Collect all sentences from retrieved chunks
    all_sentences = []
    for chunk, _ in retrieved_chunks_with_scores:
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", chunk)
        for s in sentences:
            s = s.strip()
            # Filter very short sentences (likely nav text)
            if len(s) > 40:
                all_sentences.append(s)

    print(f"  → Total candidate sentences: {len(all_sentences)}")

    if not all_sentences:
        return "No relevant sentences found."

    # Embed query and all sentences
    q_vec = embed_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    s_vecs = embed_model.encode(
        all_sentences, convert_to_numpy=True, normalize_embeddings=True
    )

    # Cosine similarity = dot product (since both are L2-normalised)
    scores = (s_vecs @ q_vec.T).flatten()

    # Pick top_n sentences by score
    top_idx = np.argsort(scores)[::-1][:top_n]

    # Re-order in original document order for a more readable answer
    top_idx_sorted = sorted(top_idx)

    print(f"\n  ✓ Top {top_n} extracted sentences:")
    answer_parts = []
    for rank, idx in enumerate(top_idx_sorted, start=1):
        print(f"\n  [{rank}] Score: {scores[idx]:.4f}")
        print(f"       \"{all_sentences[idx]}\"")
        answer_parts.append(all_sentences[idx])

    return " ".join(answer_parts)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "█"*60)
    print("  RAG PIPELINE — WITHOUT LLM (extractive retrieval)")
    print("  No API key. No internet model. Pure retrieval.")
    print("█"*60)

    raw_text = step1_load_documents(DOCS_URL)
    chunks   = step2_chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
    embed_model, index, _, bm25 = step3_embed_and_index(chunks)
    retrieved = step4_hybrid_retrieve(QUERY, embed_model, index, bm25, chunks, TOP_K)
    answer    = step5_extractive_answer(QUERY, retrieved, embed_model, TOP_SENTENCES)

    # ── Output ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL EXTRACTIVE ANSWER (no LLM used)")
    print("="*60)
    print(f"\nQ: {QUERY}\n")
    print(f"A: {answer}")

    print("\n" + "-"*60)
    print("WHY IS THIS DIFFERENT FROM THE LLM VERSION?")
    print("-"*60)
    print("""
  ✓ The answer above is COPY-PASTED text from the documentation.
    No sentence was generated or rephrased.

  ✗ It may read awkwardly — sentences were extracted from different
    parts of the document and just joined together.

  ✗ If the answer spans multiple topics, extraction may miss nuance.

  ✓ Zero hallucination risk — every word is from the source.

  ✓ No API key, no cost, no rate limit, fully offline.

  → The LLM's job in RAG is to SYNTHESISE these extracted chunks
    into a clean, fluent, human-readable answer. The retrieval
    pipeline is identical in both cases.
""")
    print("█"*60)


if __name__ == "__main__":
    main()