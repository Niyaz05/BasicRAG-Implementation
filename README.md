# 🔍 RAG (Retrieval-Augmented Generation) — Python Docs

A hands-on RAG implementation built to understand how retrieval pipelines work.  
Three versions: one with a Gemini LLM, one with a Local LLM (Gemma 4 via Ollama), and one with zero API key (extractive only).

---

## 📁 Project Structure

```
.
├── ragwithllm.py      # Full RAG pipeline → scrape → embed → retrieve → Gemini answer
├── ragwithgemma4.py   # Full RAG pipeline → scrape → embed → retrieve → Local Gemma 4 answer
├── ragwithoutllm.py      # Same pipeline, extractive answer only (no LLM, no API key)
├── .env                    # Your API key (never committed — see .gitignore)
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo and enter the folder
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. (Optional but recommended) Create a virtual environment
```bash
python -m venv venv

# Activate it:
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install requests beautifulsoup4 sentence-transformers faiss-cpu google-generativeai python-dotenv ollama
```

### 4. Set up your `.env` file
Create a `.env` file in the root of the project:
```
GOOGLE_API_KEY=your_key_here
```
Get a free key at → https://aistudio.google.com/app/apikey (no credit card needed)

### 5. (For Gemma 4) Install Ollama and pull the model
Make sure [Ollama](https://ollama.com/) is installed and running, then pull the model:
```bash
ollama pull gemma4
```

---

## 🚀 Running the files

### File 1 — RAG with Gemini LLM
```bash
python rag_with_gemini.py
```
Prints all 6 pipeline steps and a fluent generated answer.

### File 2 — RAG without any LLM
```bash
python rag_without_llm.py
```
Prints all 5 pipeline steps and an extractive answer (verbatim sentences from the docs).  
No API key needed.

### File 3 — RAG with Local Gemma 4 (Ollama)
```bash
python ragwithgemma4.py
```
Runs the full pipeline locally with zero API keys required, using Gemma 4 via Ollama.

---

## 🔄 How the pipeline works

```
[Docs URL]
    ↓  Step 1 — Scrape & clean text
[Raw Text]
    ↓  Step 2 — Split into overlapping chunks
[Chunks]
    ↓  Step 3 — Embed with all-MiniLM-L6-v2 (local model)
[Vectors]  +  BM25 index (file 2 only)
    ↓  Step 4 — Store in FAISS vector store
[FAISS Index]
    ↓  Step 5 — Embed query → find top-k similar chunks
[Retrieved Chunks]
    ↓  Step 6 — Build prompt → send to Gemini   (file 1)
              OR send to Local Gemma 4           (file 3)
              OR extract best sentences          (file 2)
[Answer]
```

---

## ⚙️ Key parameters (edit at the top of each file)

| Parameter | Default | What it controls |
|---|---|---|
| `CHUNK_SIZE` | 500 chars | Size of each text chunk |
| `CHUNK_OVERLAP` | 100 chars | Shared characters between adjacent chunks |
| `TOP_K` | 4 | Number of chunks retrieved per query |
| `TOP_SENTENCES` | 3 | Sentences extracted in the no-LLM version |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Free-tier model, fast |
| `OLLAMA_MODEL` | `gemma4` | Local model tag for Ollama |
| `OLLAMA_NUM_CTX`| 8192 | Context window override for Gemma 4 |
| `QUERY` | see file | The question asked at the end of the pipeline |

---

## 🆚 LLM vs No-LLM vs Local LLM — what's the difference?

| | `ragwithllm.py` | `ragwithgemma4.py` | `ragwithoutllm.py` |
|---|---|---|---|
| Answer style | Fluent, synthesised | Fluent, synthesised | Raw extracted sentences |
| Hallucination risk | Low (grounded in context) | Low (grounded in context) | Zero (verbatim from source) |
| Reasoning across chunks | ✅ | ✅ | ❌ |
| API key required | ✅ Yes (free) | ❌ No | ❌ No |
| Internet needed | ✅ (for scraping + API) | ✅ (for scraping only) | ✅ (for scraping only) |
| Local Compute | Low | High (runs LLM locally) | Low |

The retrieval logic is identical in both files. The LLM only handles the final step — rewriting extracted context into a clean answer.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `requests` | Fetch web pages |
| `beautifulsoup4` | Parse and clean HTML |
| `sentence-transformers` | Local embedding model (all-MiniLM-L6-v2) |
| `faiss-cpu` | Vector store and similarity search |
| `google-generativeai` | Gemini API client |
| `python-dotenv` | Load `.env` file |
| `ollama` | Python client for local LLMs |