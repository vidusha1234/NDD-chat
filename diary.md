# NDD Chat – Project Deployment Diary

A full record of what was done, what went wrong, and how each problem was solved.
Use this as a learning reference for future projects.

---

## 1. Project Overview

### What the app is
- A **RAG (Retrieval-Augmented Generation)** chatbot for parents of children with neurodevelopmental delays
- Users ask questions; the app searches a PDF document for relevant content and uses an AI to write a compassionate, informed answer
- Built with:
  - **FastAPI** (Python web framework) — handles API requests
  - **Groq API** (LLaMA 3.3-70b model) — generates the AI responses
  - **ChromaDB** (initially) — vector database for storing document embeddings
  - **sentence-transformers** (initially) — converts text into numerical vectors for semantic search
  - **HTML / CSS / JavaScript** — the chat interface (frontend)

### Project structure
```
NDD chat/
├── backend/          ← Python FastAPI app
│   ├── app.py        ← Main server, API endpoints
│   ├── rag_system.py ← Orchestrates search + AI generation
│   ├── vector_store.py ← Search index (ChromaDB → BM25)
│   ├── ai_generator.py ← Groq API calls
│   ├── config.py     ← Environment variables / settings
│   └── ...
├── frontend/         ← Static HTML/CSS/JS chat interface
├── docs/             ← PDF documents the chatbot reads from
├── render.yaml       ← Render deployment config
├── vercel.json       ← Vercel deployment config
└── pyproject.toml    ← Python dependencies
```

---

## 2. Setting Up Version Control (GitHub)

### What we did
1. Updated `.gitignore` to exclude sensitive/unnecessary files
2. Initialised a local git repository
3. Created the GitHub repository **NDD-chat** using the GitHub CLI (`gh`)
4. Made the first commit and pushed all project files

### Key concept: `.gitignore`
A `.gitignore` file tells git which files/folders to **never** upload to GitHub.

We excluded:
| Item | Why |
|---|---|
| `.env` | Contains your secret API keys — never share these |
| `.venv/` | Local Python environment (947 MB!) — not needed on GitHub |
| `backend/chroma_db/` | Auto-generated database — rebuilt on startup |
| `.claude/` | Claude Code internal files |
| `Unwanted/` | Old course scripts not needed in the repo |
| `.DS_Store` | Mac system file, irrelevant to the project |

### Key learning
Your actual project code was only ~2 MB. The 992 MB folder size was almost entirely your `.venv` (947 MB of Python packages). None of that goes to GitHub.

---

## 3. Deployment Strategy

### The plan
Host the app in two parts:
- **Frontend** (HTML/JS/CSS) → **Vercel** (specialises in static sites, always fast)
- **Backend** (FastAPI + Python) → **Render** (supports long-running Python servers)

### Why not put everything on Vercel?
Vercel runs **serverless functions** — short-lived processes that start and stop instantly. Our backend couldn't work there because:
1. **ChromaDB writes to disk** — Vercel has no persistent storage
2. **sentence-transformers loads PyTorch** — a 400+ MB model that exceeds Vercel's 50 MB function limit
3. **PDF indexing on startup** — too slow for a serverless cold start

### Config files created

**`render.yaml`** — tells Render how to build and start the backend:
```yaml
buildCommand: pip install uv && uv sync --frozen
startCommand: cd backend && uv run uvicorn app:app --host 0.0.0.0 --port $PORT
```

**`vercel.json`** — tells Vercel to serve the frontend folder as a static site:
```json
{
  "builds": [{ "src": "frontend/**", "use": "@vercel/static" }],
  "routes": [{ "src": "/(.*)", "dest": "/frontend/$1" }]
}
```

---

## 4. Render Deployment – Barriers & Solutions

### Barrier 1: Out of Memory (512 MB limit exceeded)
**Error:** `Ran out of memory (used over 512MB) while running your code`

**Cause:** `sentence-transformers` loads the full PyTorch machine learning framework to create text embeddings. PyTorch alone uses ~400 MB of RAM. Render's free tier only allows 512 MB total.

**Solution:** Replaced `sentence-transformers` (PyTorch-based) with ChromaDB's built-in **ONNX embedding function** (`DefaultEmbeddingFunction`). ONNX runs the same model but without PyTorch — much lighter.

**What was removed:** torch, transformers, CUDA packages, scikit-learn, scipy — 28+ packages gone.

---

### Barrier 2: Still Out of Memory
**Error:** `Ran out of memory (used over 512MB)` again

**Cause:** Even the ONNX approach still loaded a local model (~100 MB model + ChromaDB overhead + FastAPI + PDF processing = still over 512 MB).

**Solution:** Switched to **HuggingFace Inference API** — instead of loading any model locally, we send text to HuggingFace's servers and they return the embeddings. Zero local model loading.

**Key change in `vector_store.py`:**
```python
# Before: loads model locally (uses RAM)
self.embedding_function = DefaultEmbeddingFunction()

# After: calls external API (uses almost no RAM)
self.embedding_function = HuggingFaceEmbeddingFunction(api_key=hf_api_key, ...)
```

A free HuggingFace token (`hf_...`) was added as an environment variable on Render.

---

### Barrier 3: ChromaDB `name()` attribute error
**Error:** `AttributeError: 'HFEmbeddingFunction' object has no attribute 'name'`

**Cause:** ChromaDB 1.0 requires all custom embedding functions to have a `name()` method. Our custom class didn't have one.

**Solution:** Added `def name(self) -> str: return "hf_inference_api"` to the class.

---

### Barrier 4: ChromaDB persistent metadata conflict
**Error:** Same `name()` error persisted even after the fix

**Cause:** Render was caching old build artefacts including a `chroma_db/` folder created with a different embedding function. When the app restarted, ChromaDB found the old metadata and rejected the new embedding function as incompatible.

**Solution:** Switched from `PersistentClient` (writes to disk) to `EphemeralClient` (in-memory only). Since the app rebuilds the database from the PDF on every startup anyway, there is no reason to persist it.

```python
# Before:
self.client = chromadb.PersistentClient(path=chroma_path, ...)

# After:
self.client = chromadb.EphemeralClient(...)
```

---

### Barrier 5: HuggingFace API endpoint retired (410 Gone)
**Error:** `410 Client Error: Gone for url: https://api-inference.huggingface.co/pipeline/feature-extraction/...`

**Cause:** HuggingFace permanently shut down their `/pipeline/feature-extraction/` API endpoint.

**Attempted fix:** Changed URL to `/models/` endpoint — still returned 410. HuggingFace had restricted access to this model on their free tier entirely.

**Final solution — replaced all embedding logic with BM25 keyword search:**

BM25 is a classic information retrieval algorithm (the same underlying technique used by search engines). Instead of converting text into numerical vectors, it scores documents based on **keyword frequency and relevance**. No external API, no model, no RAM overhead.

**Result:**
- Package count dropped from **92 → 25**
- All of the following were removed: `chromadb`, `onnxruntime`, `tokenizers`, `kubernetes`, `grpcio`, all OpenTelemetry packages, `requests`, `numpy`, `sympy` and more
- Build time dropped significantly
- Memory usage well within Render's 512 MB free tier

**Key concept — why BM25 still works well here:**
The Groq LLM formulates smart search queries (e.g. translating "my child cries on the phone" into "emotional regulation strategies"). BM25 then finds the right document chunks using those keywords. The AI writes the final answer from those chunks.

---

## 5. Frontend API URL — Local vs Production

### The problem
When testing locally, the frontend needs to call `localhost:8000/api`.
When deployed on Vercel, it needs to call the Render backend URL.

### Solution: auto-detect in `script.js`
```javascript
const API_URL = window.location.hostname === 'localhost'
    ? '/api'
    : 'https://ndd-chat-an3u.onrender.com/api';
```
This automatically uses the right URL depending on where the app is running.

---

## 6. Local Development Issues

### Barrier 1: Broken `.venv` after folder rename
**Error:** `/starting-ragchatbot-codebase-main/.venv/bin/python: No such file or directory`

**Cause:** The `.venv` virtual environment stores **absolute paths** to the Python interpreter. When the project folder was renamed, those paths became invalid.

**Solution:**
```bash
rm -rf .venv    # delete the broken environment
uv sync         # recreate it fresh in the correct location
```

### Barrier 2: Port already in use
**Error:** `[Errno 48] Address already in use`

**Cause:** A previous server process was still running on port 8000.

**Solution:**
```bash
lsof -ti:8000 | xargs kill -9   # find and kill whatever is using port 8000
```

---

## 7. Vercel Deployment

### Barrier: Vercel detected the app as FastAPI
**Problem:** Vercel automatically detected Python files and tried to deploy the backend as a serverless FastAPI app — not what we wanted.

**Solution:** In the Vercel import screen:
- Changed **Application Preset** from `FastAPI` → `Other`
- Changed **Root Directory** to `frontend`

This told Vercel to ignore the Python backend entirely and just serve the static HTML/JS/CSS files.

---

## 8. Final Architecture

```
User's browser
      │
      ▼
  Vercel (frontend)
  vercel.app URL
  Serves: index.html, script.js, style.css
      │
      │  API calls to Render backend
      ▼
  Render (backend)
  onrender.com URL
  Runs: FastAPI + BM25 search + Groq AI
      │
      │  Reads PDF on startup
      ▼
  docs/Challenging behvior.pdf
  (indexed into BM25 at startup)
      │
      │  AI generation
      ▼
  Groq API (LLaMA 3.3-70b)
```

---

## 9. Deployment Workflow Going Forward

Both Render and Vercel are connected to your GitHub repository and **auto-deploy on every push**.

```
Make changes locally
        ↓
Test at localhost:8000
        ↓
git add .
git commit -m "describe your change"
git push
        ↓
Render redeploys backend automatically
Vercel redeploys frontend automatically
```

### One thing to note about Render free tier
Render's free tier **spins down the server after 15 minutes of inactivity**. The first request after a period of no use will take ~30 seconds while the server wakes up and re-indexes the PDF. This is normal — just open the app yourself before sharing the link with someone important.

---

## 10. Key Technologies Learned

| Technology | What it does |
|---|---|
| **Git / GitHub** | Version control — tracks all changes, enables collaboration |
| **Vercel** | Hosts static frontends (HTML/CSS/JS), globally fast |
| **Render** | Hosts backend Python servers, free tier available |
| **FastAPI** | Python framework for building APIs quickly |
| **BM25** | Keyword-based search algorithm — finds relevant document chunks |
| **RAG** | Retrieval-Augmented Generation — search documents, then use AI to answer |
| **Groq** | Fast LLM API using LLaMA models |
| **uv** | Fast Python package manager (replaces pip) |
| **`.env` files** | Store secret keys locally, never commit to GitHub |
