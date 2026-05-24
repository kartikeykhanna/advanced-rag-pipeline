# 🧠 Advanced Production RAG Pipeline
### Technical Academic Research & Financial Intelligence Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green?logo=chainlink)](https://www.langchain.com/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-Free%20Tier-orange?logo=google)](https://aistudio.google.com/)
[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-yellow?logo=googlecolab)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Overview

A **production-grade, 6-stage Retrieval-Augmented Generation (RAG) pipeline** built entirely on **free, open-source tools** — no paid API keys required beyond a free Google AI Studio key.

Designed for dense, complex documents: research papers, financial reports, technical PDFs with formulas, nested headers, and exact numeric metrics.

### The Problem This Solves

| Native RAG Failure | Solution Implemented |
|---|---|
| 🔴 Hallucinations | Strict anti-hallucination system prompt + mandatory source citation |
| 🔴 Lost-in-the-middle context neglect | Section breadcrumb injected into every chunk + MMR retrieval |
| 🔴 Irrelevant chunking | Two-pass structural split (Markdown headers → token budget) |
| 🔴 Token waste | CrossEncoder reranking (25 → 5 chunks) + local keyword compression |
| 🔴 Keyword vs semantic mismatch | Hybrid retrieval: Vector Search (0.6) + BM25 (0.4) ensemble |

---

## 🏗️ Architecture — 6 Stages

```
PDF / Markdown
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1: Context-Aware Chunking                    │
│  MarkdownHeaderTextSplitter → RecursiveCharacter    │
│  + Section breadcrumb injected into every chunk     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2: BGE Embeddings + Chroma Vector Store      │
│  BAAI/bge-large-en-v1.5  (local, free, 1.2 GB)     │
│  Persistent Chroma DB on disk                       │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
┌─────────────────┐   ┌─────────────────┐
│  Vector Search  │   │   BM25 Search   │
│  (MMR, k=25)    │   │   (keywords)    │
│  weight = 0.6   │   │   weight = 0.4  │
└────────┬────────┘   └────────┬────────┘
         └─────────┬───────────┘
                   ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3: Hybrid Ensemble Retrieval                 │
│  25 deduplicated candidate chunks                   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 4: Cross-Encoder Re-ranking                  │
│  BAAI/bge-reranker-large  (local, free, 1.1 GB)    │
│  25 candidates → Top 5 by joint attention score     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 5: Local Compression + Gemini Synthesis      │
│  Keyword compressor (0 API calls)                   │
│  → gemini-2.5-flash with anti-hallucination prompt  │
│  → Inline source citations [Source: file, Page N]   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 6: Ragas Evaluation                          │
│  faithfulness · answer_relevancy                    │
│  context_recall · context_precision                 │
│  Judge: gemini-2.5-flash (free)                     │
└─────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Tool | Cost |
|---|---|---|
| LLM (Synthesis + Evaluation) | Google Gemini 2.5 Flash | ✅ Free |
| Embeddings (Indexing) | BAAI/bge-large-en-v1.5 | ✅ Free (local) |
| Re-ranker | BAAI/bge-reranker-large | ✅ Free (local) |
| Vector Store | ChromaDB (persistent) | ✅ Free (local) |
| Keyword Retrieval | BM25 (rank-bm25) | ✅ Free (local) |
| Orchestration | LangChain v0.2+ | ✅ Free |
| Evaluation | Ragas | ✅ Free |
| Runtime | Google Colab | ✅ Free |

**Total API cost: $0.00**

---

## 🚀 Quick Start

### Option A — Run on Google Colab (Recommended)

1. Open the notebook: [`Advanced_RAG_Pipeline.ipynb`](Advanced_RAG_Pipeline.ipynb)
2. Click **"Open in Colab"** badge at the top of the notebook
3. Get your free Google API key (see below)
4. Run all cells in order

### Option B — Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/advanced-rag-pipeline.git
cd advanced-rag-pipeline
pip install -r requirements.txt
jupyter notebook Advanced_RAG_Pipeline.ipynb
```

---

## 🔑 Getting Your Free Google API Key

> No credit card required. Takes 2 minutes.

| Step | Action |
|---|---|
| 1 | Go to **https://aistudio.google.com** |
| 2 | Sign in with any Google / Gmail account |
| 3 | Accept Terms of Service |
| 4 | Left sidebar → click **"Get API key"** |
| 5 | Click **"+ Create API key"** |
| 6 | Copy the key — it starts with `AIza...` |
| 7 | In the notebook Cell 1, set: `os.environ["GOOGLE_API_KEY"] = "AIza..."` |

**Free tier limits:**
- Gemini 2.5 Flash: 5 requests/min · 500 requests/day
- No billing required for free tier

---

## 📁 Repository Structure

```
advanced-rag-pipeline/
│
├── Advanced_RAG_Pipeline.ipynb   # Main Colab notebook (all 6 stages)
├── requirements.txt              # All Python dependencies
├── .gitignore                    # Excludes API keys, model cache, DB files
├── README.md                     # This file
└── sample_docs/
    └── demo_document.md          # Sample quantum finance document for testing
```

---

## 📊 Sample Output

```
======================================================================
QUESTION:
  What is the mathematical formula for 99% VaR and CVaR?

ANSWER:
  Based on the provided documents:

  • The 99% VaR formula is: VaR₀.₉₉ = μ − z₀.₉₉ × σ
    For μ=0.05, σ=0.12: VaR₀.₉₉ = 0.05 − 2.326 × 0.12 = −0.229
    [Source: demo_document.md, Page 1]

  • CVaR₀.₉₉ ≈ 26.1% for this portfolio [Source: demo_document.md, Page 1]

  • Quantum amplitude estimation delivers O(1/ε) queries vs O(N) classical
    [Source: demo_document.md, Page 1]

  Summary: The portfolio faces a 22.9% loss at 99% VaR, with quantum
  methods providing quadratic speedup over classical Monte Carlo.
======================================================================

RAGAS EVALUATION SCORECARD
======================================================================
  faithfulness           0.9500  |███████████████████ |
  answer_relevancy       0.9200  |██████████████████  |
  context_recall         0.8800  |█████████████████   |
  context_precision      0.9100  |██████████████████  |
======================================================================
```

---

## ⚙️ Key Design Decisions

### Why local embeddings instead of API embeddings?
`BAAI/bge-large-en-v1.5` consistently ranks in the top tier on the MTEB leaderboard. Running it locally means zero cost, no rate limits, and no data leaving your machine.

### Why CrossEncoder reranking?
Bi-encoder similarity (cosine) scores query and document independently. CrossEncoder reads both together, capturing nuanced relevance signals — critical for dense technical text where many passages look superficially similar.

### Why remove LLMChainExtractor?
On the free tier, LLMChainExtractor burns 5 Gemini calls per query (one per re-ranked chunk). After CrossEncoder reranking, chunks are already highly relevant — a local keyword compressor achieves equivalent noise reduction at zero API cost.

### Why Hybrid Retrieval?
Vector search misses exact matches: a query for `VaR₀.₉₉` may not match semantically but BM25 will catch the exact token. Ensemble (0.6 vector + 0.4 BM25) captures both conceptual and lexical relevance.

---

## 🐛 Known Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ValidationError: query_instruction extra fields not permitted` | langchain-huggingface ≥ 0.1 removed this param | Move to `encode_kwargs={"prompt": "..."}` |
| `429 ResourceExhausted limit: 0` | Daily free quota exhausted | Use local compressor (zero API calls) |
| `InternalError: readonly database` | Colab filesystem permissions | Set `CHROMA_PERSIST_DIR = "/tmp/chroma_db"` |
| `ModuleNotFoundError: langchain.retrievers` | LangChain v0.2 split packages | Use `langchain_community.retrievers.ensemble` |
| `NameError: CrossEncoder not defined` | Missing import cell | Run imports cell before class definitions |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [BAAI](https://huggingface.co/BAAI) for bge-large and bge-reranker models
- [LangChain](https://langchain.com) for the orchestration framework
- [Google AI Studio](https://aistudio.google.com) for the free Gemini API
- [Ragas](https://ragas.io) for the RAG evaluation framework
- [ChromaDB](https://www.trychroma.com) for the local vector store
