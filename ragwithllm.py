"""
rag_with_gemini.py
==================
A basic RAG (Retrieval-Augmented Generation) pipeline that:
  1. Scrapes Python documentation from the web
  2. Chunks the text
  3. Embeds chunks using a free local sentence-transformer model
  4. Stores embeddings in FAISS (in-memory vector store)
  5. Retrieves the most relevant chunks for a query
  6. Sends query + context to Google Gemini (free tier) for answer generation

SETUP (run once in your terminal):
    pip install requests beautifulsoup4 sentence-transformers faiss-cpu google-generativeai

GET A FREE GEMINI API KEY:
    → https://aistudio.google.com/app/apikey  (free, no credit card needed)
    → Set it below in GEMINI_API_KEY
"""

import re
import os
from dotenv import load_dotenv
import requests
import numpy as np
from bs4 import BeautifulSoup
import google as genai

# ─────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")   # paste your free key here
GEMINI_MODEL = "gemini-2.0-flash-001"        # free-tier model; fast and capable

DOCS_URL       = "https://docs.python.org/3/tutorial/index.html"  # source page

CHUNK_SIZE     = 500    # characters per chunk (not tokens; simpler for a tutorial)
CHUNK_OVERLAP  = 100    # characters of overlap between adjacent chunks

TOP_K          = 4      # how many chunks to retrieve and give to the LLM

QUERY          = "What is Python and how is it useful?"  # the question we ask


# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DOCUMENT
# Fetch the Python tutorial index page and all linked sub-pages,
# then extract clean text (strip HTML tags, scripts, nav, etc.)
# ─────────────────────────────────────────────────────────────────
def step1_load_documents(base_url: str) -> str:
    print("\n" + "="*60)
    print("STEP 1: LOADING DOCUMENTS")
    print("="*60)
    print(f"  → Fetching base page: {base_url}")

    headers = {"User-Agent": "Mozilla/5.0 (educational RAG demo)"}
    resp = requests.get(base_url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Collect links to sub-pages from the tutorial index
    base = "https://docs.python.org/3/tutorial/"
    sub_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # only local tutorial pages, skip external links and anchors
        if href.endswith(".html") and not href.startswith("http"):
            sub_links.append(base + href)

    # Keep first 6 sub-pages so the demo stays fast
    sub_links = list(dict.fromkeys(sub_links))[:6]
    print(f"  → Found {len(sub_links)} sub-pages to scrape")

    all_text = ""

    # Also extract text from the index page itself
    pages_to_scrape = [base_url] + sub_links

    for url in pages_to_scrape:
        print(f"  → Scraping: {url}")
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            page_soup = BeautifulSoup(r.text, "html.parser")

            # Remove nav, header, footer, script, style — we only want content
            for tag in page_soup(["nav", "header", "footer", "script", "style", "aside"]):
                tag.decompose()

            # The main documentation body is in <div role="main"> or <div class="body">
            main = (page_soup.find("div", role="main")
                    or page_soup.find("div", class_="body")
                    or page_soup.find("body"))

            text = main.get_text(separator=" ", strip=True) if main else ""

            # Collapse multiple whitespace
            text = re.sub(r"\s+", " ", text).strip()
            all_text += "\n\n" + text

        except Exception as e:
            print(f"    ⚠ Could not scrape {url}: {e}")

    print(f"\n  ✓ Total text loaded: {len(all_text):,} characters")
    return all_text.strip()


# ─────────────────────────────────────────────────────────────────
# STEP 2 — CHUNK TEXT
# Split the big text blob into smaller overlapping pieces.
# Smaller pieces embed better (one topic per chunk = better cosine
# similarity match when searching).
# ─────────────────────────────────────────────────────────────────
def step2_chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    print("\n" + "="*60)
    print("STEP 2: CHUNKING TEXT")
    print("="*60)
    print(f"  → chunk_size  : {chunk_size} chars")
    print(f"  → chunk_overlap: {overlap} chars")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        # Only keep chunks that are long enough to be meaningful
        if len(chunk) > 50:
            chunks.append(chunk)

        # Slide forward by (chunk_size - overlap) to create overlap
        start += chunk_size - overlap

    print(f"  ✓ Total chunks created: {len(chunks)}")
    print(f"  ✓ Example chunk (first 200 chars):")
    print(f"    \"{chunks[0][:200]}...\"")
    return chunks


# ─────────────────────────────────────────────────────────────────
# STEP 3 — EMBED CHUNKS
# Convert each text chunk into a float vector using a local
# sentence-transformer model (runs fully offline, no API key).
# Similar texts → vectors that are close in space.
# ─────────────────────────────────────────────────────────────────
def step3_embed_chunks(chunks: list[str]) -> tuple:
    print("\n" + "="*60)
    print("STEP 3: EMBEDDING CHUNKS (local model, no API key needed)")
    print("="*60)

    # Import here so the error is clear if not installed
    from sentence_transformers import SentenceTransformer

    # all-MiniLM-L6-v2: tiny (80 MB), fast, good quality for English
    model_name = "all-MiniLM-L6-v2"
    print(f"  → Loading embedding model: {model_name}")
    embed_model = SentenceTransformer(model_name)

    print(f"  → Embedding {len(chunks)} chunks... (may take 10–30s first run)")
    # encode() returns a numpy array of shape (num_chunks, embedding_dim)
    embeddings = embed_model.encode(
        chunks,
        show_progress_bar=True,  # prints a nice progress bar
        convert_to_numpy=True,
        normalize_embeddings=True  # L2-normalize for cosine similarity via dot product
    )

    print(f"  ✓ Embedding shape : {embeddings.shape}")
    print(f"  ✓ Vector dimension: {embeddings.shape[1]}")
    return embed_model, embeddings


# ─────────────────────────────────────────────────────────────────
# STEP 4 — BUILD VECTOR STORE (FAISS)
# Store all chunk vectors in a FAISS index so we can search
# millions of vectors in milliseconds using approximate nearest
# neighbour (ANN) search.
# ─────────────────────────────────────────────────────────────────
def step4_build_vector_store(embeddings: np.ndarray):
    print("\n" + "="*60)
    print("STEP 4: BUILDING VECTOR STORE (FAISS)")
    print("="*60)

    import faiss  # installed via faiss-cpu

    dim = embeddings.shape[1]  # embedding dimension (384 for MiniLM)

    # IndexFlatIP = exact inner-product search (works as cosine similarity
    # because we normalised the vectors in Step 3)
    index = faiss.IndexFlatIP(dim)

    # Add all chunk vectors to the index
    index.add(embeddings.astype("float32"))

    print(f"  ✓ FAISS index type : IndexFlatIP (exact cosine search)")
    print(f"  ✓ Vectors stored   : {index.ntotal}")
    print(f"  ✓ Vector dimension : {dim}")
    return index


# ─────────────────────────────────────────────────────────────────
# STEP 5 — RETRIEVE RELEVANT CHUNKS
# Embed the user query with the same model, then find the top-k
# chunks whose vectors are most similar (highest cosine score).
# ─────────────────────────────────────────────────────────────────
def step5_retrieve(query: str, embed_model, index, chunks: list[str], top_k: int) -> list[str]:
    print("\n" + "="*60)
    print("STEP 5: RETRIEVING RELEVANT CHUNKS")
    print("="*60)
    print(f"  → Query  : \"{query}\"")
    print(f"  → top_k  : {top_k}")

    # Embed the query — must use the SAME model used to embed chunks
    query_vec = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # FAISS search returns (scores, indices) for top_k nearest vectors
    scores, indices = index.search(query_vec, top_k)

    print(f"\n  ✓ Top {top_k} retrieved chunks:")
    retrieved = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        chunk_preview = chunks[idx][:120].replace("\n", " ")
        print(f"\n  [{rank}] Score: {score:.4f}")
        print(f"       Chunk #{idx}: \"{chunk_preview}...\"")
        retrieved.append(chunks[idx])

    return retrieved


# ─────────────────────────────────────────────────────────────────
# STEP 6 — GENERATE ANSWER WITH GEMINI
# Build a prompt that contains the retrieved chunks as "context",
# then ask Gemini to answer ONLY from that context.
# This is the "Augmented Generation" part of RAG.
# ─────────────────────────────────────────────────────────────────
from google import genai

def step6_generate_answer(query, retrieved_chunks, api_key, model_name):


    print("\n============================================================")
    print("STEP 6: GENERATING ANSWER WITH GEMINI")
    print("============================================================")

    print(f"  → Model: {model_name}")

    # Combine retrieved chunks
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    ```

    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {query}
    """


    print(f"  → Prompt length: {len(prompt):,} characters")
    print("  → Sending to Gemini...")

    # Create Gemini client
    client = genai.Client(api_key=api_key)

    # Generate response
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    return response.text




# ─────────────────────────────────────────────────────────────────
# MAIN — runs the full pipeline end to end
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "█"*60)
    print("  RAG PIPELINE — Python Docs + Gemini (free tier)")
    print("█"*60)

    # ── Step 1: Load ──────────────────────────────────────────────
    raw_text = step1_load_documents(DOCS_URL)

    # ── Step 2: Chunk ─────────────────────────────────────────────
    chunks = step2_chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)

    # ── Step 3: Embed ─────────────────────────────────────────────
    embed_model, embeddings = step3_embed_chunks(chunks)

    # ── Step 4: Store ─────────────────────────────────────────────
    index = step4_build_vector_store(embeddings)

    # ── Step 5: Retrieve ──────────────────────────────────────────
    retrieved_chunks = step5_retrieve(QUERY, embed_model, index, chunks, TOP_K)

    # ── Step 6: Generate ──────────────────────────────────────────
    answer = step6_generate_answer(QUERY, retrieved_chunks, GEMINI_API_KEY, GEMINI_MODEL)

    # ── Final answer ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL ANSWER")
    print("="*60)
    print(f"\nQ: {QUERY}\n")
    print(f"A: {answer}")
    print("\n" + "█"*60)


if __name__ == "__main__":
    main()