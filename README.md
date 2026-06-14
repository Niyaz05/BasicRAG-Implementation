# 🔍 RAG (Retrieval-Augmented Generation) — Python Docs

A hands-on RAG implementation built to understand how retrieval pipelines work.  
Two versions: one with a Gemini LLM, one with zero API key.

---

## 📁 Project Structure

```
.
├── ragwithllm.py      # Full RAG pipeline → scrape → embed → retrieve → Gemini answer
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
pip install requests beautifulsoup4 sentence-transformers faiss-cpu google-generativeai python-dotenv
```

### 4. Set up your `.env` file
Create a `.env` file in the root of the project:
```
GOOGLE_API_KEY=your_key_here
```
Get a free key at → https://aistudio.google.com/app/apikey (no credit card needed)

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
| `QUERY` | see file | The question asked at the end of the pipeline |

---

## 🆚 LLM vs No-LLM — what's the difference?

| | `rag_with_gemini.py` | `rag_without_llm.py` |
|---|---|---|
| Answer style | Fluent, synthesised | Raw extracted sentences |
| Hallucination risk | Low (grounded in context) | Zero (verbatim from source) |
| Reasoning across chunks | ✅ | ❌ |
| API key required | ✅ Yes (free) | ❌ No |
| Internet needed | ✅ (for scraping + API) | ✅ (for scraping only) |

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