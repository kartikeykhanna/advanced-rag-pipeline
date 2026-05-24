# =============================================================================
# ADVANCED PRODUCTION RAG PIPELINE
# Target: Technical Academic Research & Financial Intelligence Assistant
# =============================================================================
#
# INSTALLATION (run once before executing this script):
#
# pip install langchain langchain-community langchain-anthropic langchain-huggingface
# pip install chromadb sentence-transformers rank-bm25 transformers torch
# pip install pypdf unstructured ragas datasets anthropic openai
# pip install accelerate einops huggingface_hub
#
# ENVIRONMENT VARIABLES REQUIRED:
#   ANTHROPIC_API_KEY  — for Claude LLM synthesis
#   OPENAI_API_KEY     — used by Ragas evaluation framework internally
#
# =============================================================================

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── LangChain core ────────────────────────────────────────────────────────────
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# ── Re-ranking ────────────────────────────────────────────────────────────────
from sentence_transformers import CrossEncoder

# ── Document loading ──────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader

# ── Evaluation ────────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

import pandas as pd

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AdvancedRAGPipeline")


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

EMBEDDING_MODEL_NAME   = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL_NAME    = "BAAI/bge-reranker-large"
LLM_MODEL_NAME         = "claude-3-5-sonnet-20241022"
CHROMA_PERSIST_DIR     = "./chroma_db"
COLLECTION_NAME        = "rag_research_docs"
ENSEMBLE_VECTOR_WEIGHT = 0.6
ENSEMBLE_BM25_WEIGHT   = 0.4
CANDIDATE_K            = 25   # chunks retrieved before re-ranking
RERANKED_TOP_K         = 5    # chunks kept after re-ranking

SYSTEM_PROMPT = """You are an expert Technical Academic Research and Financial Intelligence Assistant.

STRICT OPERATING RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:

1. ANTI-HALLUCINATION: You may ONLY state facts that are explicitly present in the provided context
   passages. If the context does not contain sufficient information to answer the question, you must
   say: "The provided documents do not contain enough information to answer this question."

2. SOURCE CITATION: Every factual claim in your answer MUST be followed by an inline citation
   referencing the source. Use the format [Source: <filename>, Page <page_number>] if page metadata
   is available, otherwise use [Source: <source>].

3. TECHNICAL PRECISION: Preserve exact values — numbers, formulas, asset prices, model names,
   code identifiers — verbatim from the source. Never paraphrase numerical or technical data.

4. STRUCTURE: Provide a clear, well-structured answer. Use bullet points or numbered lists for
   multi-part answers. Conclude with a brief "Summary" sentence.

5. UNCERTAINTY: If you are uncertain about any detail, explicitly flag it with "(unverified)".

Context passages:
{context}
"""

HUMAN_PROMPT = "Question: {question}"

# =============================================================================
# STAGE 1 — CONTEXT-AWARE CHUNKING
# =============================================================================

class DocumentIngestionEngine:
    """
    Loads PDF or Markdown documents, splits them using structure-aware
    chunking that preserves header hierarchy as chunk metadata, and returns
    a flat list of LangChain Document objects ready for embedding.
    """

    # Markdown header levels to use as semantic boundaries
    MARKDOWN_HEADERS = [
        ("#",    "header_1"),
        ("##",   "header_2"),
        ("###",  "header_3"),
        ("####", "header_4"),
    ]

    # Fallback character boundaries for plain text / extracted PDF text
    RECURSIVE_SEPARATORS = [
        "\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""
    ]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.MARKDOWN_HEADERS,
            strip_headers=False,          # keep headers inside the chunk text
            return_each_line=False,
        )
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            separators=self.RECURSIVE_SEPARATORS,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def ingest(self, file_path: str) -> List[Document]:
        """
        Accepts a PDF or Markdown file path.
        Returns enriched, header-aware Document chunks.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        suffix = path.suffix.lower()
        logger.info("Loading document: %s (type=%s)", path.name, suffix)

        if suffix == ".pdf":
            raw_docs = self._load_pdf(path)
        elif suffix in {".md", ".markdown"}:
            raw_docs = self._load_markdown(path)
        else:
            raise ValueError(f"Unsupported file type '{suffix}'. Supported: .pdf, .md")

        chunks = self._split_documents(raw_docs, source_name=path.name)
        logger.info("Produced %d enriched chunks from '%s'.", len(chunks), path.name)
        return chunks

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_pdf(self, path: Path) -> List[Document]:
        loader = PyPDFLoader(str(path))
        pages  = loader.load()
        # Annotate source metadata on every page
        for page in pages:
            page.metadata.setdefault("source", path.name)
        return pages

    def _load_markdown(self, path: Path) -> List[Document]:
        loader = UnstructuredMarkdownLoader(str(path), mode="elements")
        docs   = loader.load()
        for doc in docs:
            doc.metadata.setdefault("source", path.name)
        return docs

    def _split_documents(self, docs: List[Document], source_name: str) -> List[Document]:
        """
        Two-pass split:
          Pass 1 — MarkdownHeaderTextSplitter to capture semantic structure.
          Pass 2 — RecursiveCharacterTextSplitter to enforce max token budget
                   while propagating header metadata into every sub-chunk.
        """
        all_chunks: List[Document] = []

        for doc in docs:
            # Pass 1: structural split on Markdown headers
            md_chunks = self._md_splitter.split_text(doc.page_content)

            for md_chunk in md_chunks:
                # Merge original page-level metadata with header metadata
                merged_meta = {**doc.metadata, **md_chunk.metadata}
                merged_meta["source"] = source_name

                # Build a breadcrumb string for the LLM to understand provenance
                breadcrumb_parts = [
                    merged_meta.get("header_1", ""),
                    merged_meta.get("header_2", ""),
                    merged_meta.get("header_3", ""),
                    merged_meta.get("header_4", ""),
                ]
                breadcrumb = " > ".join(p for p in breadcrumb_parts if p).strip()
                if breadcrumb:
                    merged_meta["section_path"] = breadcrumb

                # Pass 2: enforce chunk size budget
                sub_docs = self._recursive_splitter.create_documents(
                    texts=[md_chunk.page_content],
                    metadatas=[merged_meta],
                )

                # Propagate breadcrumb into text prefix so LLM always sees context
                for sub_doc in sub_docs:
                    if breadcrumb and not sub_doc.page_content.startswith(breadcrumb):
                        sub_doc.page_content = f"[Section: {breadcrumb}]\n\n{sub_doc.page_content}"
                    all_chunks.append(sub_doc)

        return all_chunks


# =============================================================================
# STAGE 2 — BGE EMBEDDINGS & PERSISTENT VECTOR STORE
# =============================================================================

class VectorStoreManager:
    """
    Manages the Chroma vector store backed by BAAI/bge-large-en-v1.5 embeddings.
    Supports incremental indexing, collection reset, and retriever creation.
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, collection: str = COLLECTION_NAME):
        self.persist_dir = persist_dir
        self.collection  = collection
        self.embeddings  = self._init_embeddings()
        self._store: Optional[Chroma] = None

    # ── Embedding initialisation ──────────────────────────────────────────────

    @staticmethod
    def _init_embeddings() -> HuggingFaceEmbeddings:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        # BGE models expect a query instruction prefix for retrieval tasks
        encode_kwargs = {"normalize_embeddings": True}
        model_kwargs  = {"device": "cpu"}   # change to "cuda" if GPU is available

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            # Prepend the BGE recommended query instruction
            query_instruction="Represent this sentence for searching relevant passages: ",
        )
        logger.info("Embedding model loaded.")
        return embeddings

    # ── Store management ──────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """Index new documents into the persistent Chroma store."""
        logger.info("Indexing %d chunks into Chroma collection '%s'…", len(documents), self.collection)
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
        self._store.add_documents(documents)
        logger.info("Indexing complete. Store persisted at '%s'.", self.persist_dir)

    def load_existing(self) -> bool:
        """
        Load a previously persisted store.
        Returns True if a store with documents exists, False otherwise.
        """
        store_path = Path(self.persist_dir)
        if not store_path.exists():
            return False
        self._store = Chroma(
            collection_name=self.collection,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        count = self._store._collection.count()
        if count == 0:
            logger.warning("Chroma store found but contains 0 documents.")
            return False
        logger.info("Loaded existing Chroma store (%d chunks).", count)
        return True

    def get_vector_retriever(self, k: int = CANDIDATE_K):
        if self._store is None:
            raise RuntimeError("Vector store not initialised. Call add_documents() or load_existing() first.")
        return self._store.as_retriever(
            search_type="mmr",                  # Maximal Marginal Relevance reduces redundancy
            search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7},
        )

    def get_all_documents(self) -> List[Document]:
        """Retrieve every document in the store (needed for BM25 initialisation)."""
        if self._store is None:
            raise RuntimeError("Vector store not initialised.")
        raw = self._store._collection.get(include=["documents", "metadatas"])
        docs = []
        for text, meta in zip(raw["documents"], raw["metadatas"]):
            docs.append(Document(page_content=text, metadata=meta or {}))
        return docs


# =============================================================================
# STAGE 3 — HYBRID RETRIEVAL (Vector + BM25 Ensemble)
# =============================================================================

class HybridRetriever:
    """
    Combines semantic vector search with BM25 keyword retrieval via
    LangChain's EnsembleRetriever. Catches both conceptual and exact
    technical term matches (formulas, tickers, identifiers).
    """

    def __init__(
        self,
        vector_retriever,
        corpus_documents: List[Document],
        vector_weight: float = ENSEMBLE_VECTOR_WEIGHT,
        bm25_weight: float   = ENSEMBLE_BM25_WEIGHT,
        k: int               = CANDIDATE_K,
    ):
        if abs(vector_weight + bm25_weight - 1.0) > 1e-6:
            raise ValueError("Ensemble weights must sum to 1.0")

        logger.info("Initialising BM25 retriever over %d documents…", len(corpus_documents))
        bm25_retriever = BM25Retriever.from_documents(corpus_documents)
        bm25_retriever.k = k

        self._ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[vector_weight, bm25_weight],
        )
        self._k = k
        logger.info(
            "HybridRetriever ready (vector=%.1f, bm25=%.1f, k=%d).",
            vector_weight, bm25_weight, k,
        )

    def retrieve(self, query: str) -> List[Document]:
        results = self._ensemble.invoke(query)
        # Deduplicate by content hash (EnsembleRetriever can surface duplicates)
        seen, unique = set(), []
        for doc in results:
            key = hash(doc.page_content[:200])
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        logger.info("HybridRetriever returned %d unique candidate chunks.", len(unique))
        return unique[: self._k]


# =============================================================================
# STAGE 4 — CROSS-ENCODER RE-RANKING
# =============================================================================

class CrossEncoderReranker:
    """
    Uses BAAI/bge-reranker-large to score every (query, chunk) pair and
    returns the top-k most relevant documents with their rerank scores
    stored in metadata.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME, top_k: int = RERANKED_TOP_K):
        logger.info("Loading CrossEncoder reranker: %s", model_name)
        self._model = CrossEncoder(model_name, max_length=512)
        self._top_k = top_k
        logger.info("CrossEncoder loaded.")

    def rerank(self, query: str, candidates: List[Document]) -> List[Document]:
        if not candidates:
            return []

        pairs  = [[query, doc.page_content] for doc in candidates]
        scores = self._model.predict(pairs)

        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top    = scored[: self._top_k]

        reranked_docs = []
        for rank, (score, doc) in enumerate(top, start=1):
            doc.metadata["rerank_score"] = round(float(score), 4)
            doc.metadata["rerank_rank"]  = rank
            reranked_docs.append(doc)

        logger.info(
            "Re-ranked %d → %d docs. Top score: %.4f",
            len(candidates), len(reranked_docs), top[0][0],
        )
        return reranked_docs


# =============================================================================
# STAGE 5 — CONTEXT COMPRESSION & LLM SYNTHESIS
# =============================================================================

class SynthesisEngine:
    """
    1. Compresses the top-k re-ranked chunks with LLMChainExtractor
       (strips irrelevant sentences, keeping only query-relevant content).
    2. Feeds the compressed context to Claude with a strict anti-hallucination
       system prompt and returns the final answer with source metadata.
    """

    def __init__(self):
        self._llm = self._init_llm()
        self._compressor = LLMChainExtractor.from_llm(self._llm)
        self._chain      = self._build_chain()

    # ── LLM initialisation ────────────────────────────────────────────────────

    @staticmethod
    def _init_llm() -> ChatAnthropic:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it before running: export ANTHROPIC_API_KEY='sk-ant-...'"
            )
        logger.info("Initialising LLM: %s", LLM_MODEL_NAME)
        return ChatAnthropic(
            model=LLM_MODEL_NAME,
            anthropic_api_key=api_key,
            temperature=0.0,          # zero temperature for factual, deterministic output
            max_tokens=2048,
        )

    # ── Context compression ───────────────────────────────────────────────────

    def compress(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Applies LLMChainExtractor to each document individually, extracting
        only the sentences directly relevant to the query.
        """
        compressed: List[Document] = []
        for doc in documents:
            try:
                result = self._compressor.compress_documents([doc], query)
                if result:
                    compressed.extend(result)
                else:
                    # If compressor returns nothing, keep original (avoid data loss)
                    compressed.append(doc)
            except Exception as exc:
                logger.warning("Compression failed for a chunk (%s). Keeping original.", exc)
                compressed.append(doc)
        logger.info("Compressed %d docs → %d non-empty fragments.", len(documents), len(compressed))
        return compressed

    # ── Synthesis chain ───────────────────────────────────────────────────────

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",  HUMAN_PROMPT),
        ])
        return prompt | self._llm | StrOutputParser()

    @staticmethod
    def _format_context(documents: List[Document]) -> str:
        """
        Formats compressed documents into a numbered, citation-ready context block.
        Each passage includes its source file, page (if available), section path,
        and rerank score so the LLM can construct inline citations.
        """
        parts = []
        for i, doc in enumerate(documents, start=1):
            meta   = doc.metadata
            source = meta.get("source", "unknown")
            page   = meta.get("page", meta.get("page_number", "?"))
            section= meta.get("section_path", "")
            score  = meta.get("rerank_score", "?")

            header = f"[Passage {i}] Source: {source} | Page: {page}"
            if section:
                header += f" | Section: {section}"
            header += f" | Relevance Score: {score}"

            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n" + "\n\n---\n\n".join(parts) + "\n"

    def synthesize(self, query: str, documents: List[Document]) -> Tuple[str, List[Document]]:
        """
        Compresses documents, builds the context, invokes the LLM,
        and returns (answer_text, compressed_documents).
        """
        compressed = self.compress(query, documents)
        context    = self._format_context(compressed)
        logger.info("Invoking LLM for synthesis…")
        answer = self._chain.invoke({"context": context, "question": query})
        return answer, compressed


# =============================================================================
# STAGE 6 — RAGAS EVALUATION FRAMEWORK
# =============================================================================

class RagasEvaluator:
    """
    Wraps the Ragas evaluation library to compute four core RAG metrics:
      • faithfulness       — are claims grounded in retrieved context?
      • answer_relevancy   — does the answer address the question?
      • context_recall     — does the retrieved context cover the ground truth?
      • context_precision  — is the retrieved context free of noise?

    Accepts a list of evaluation samples; each sample is a dict with keys:
      question, answer, contexts (List[str]), ground_truth
    """

    METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]

    def evaluate(self, samples: List[dict]) -> pd.DataFrame:
        """
        Runs Ragas evaluation over the provided sample list.
        Returns a Pandas DataFrame with per-sample and aggregate scores.
        """
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is required by Ragas for its judge LLM. "
                "Export it before running: export OPENAI_API_KEY='sk-...'"
            )

        logger.info("Running Ragas evaluation on %d sample(s)…", len(samples))

        dataset = Dataset.from_dict({
            "question":    [s["question"]    for s in samples],
            "answer":      [s["answer"]      for s in samples],
            "contexts":    [s["contexts"]    for s in samples],
            "ground_truth":[s["ground_truth"]for s in samples],
        })

        results = evaluate(dataset=dataset, metrics=self.METRICS)
        df = results.to_pandas()
        logger.info("Ragas evaluation complete.")
        return df


# =============================================================================
# ORCHESTRATION CLASS
# =============================================================================

class AdvancedRAGPipeline:
    """
    Top-level orchestrator that composes all six stages into a single
    coherent pipeline. Exposes two public methods:
      • ingest(file_path)         — load, chunk, embed, and index a document
      • query(question) → dict    — run the full retrieval-to-answer pipeline
    Plus a standalone evaluation helper:
      • evaluate(samples) → DataFrame
    """

    def __init__(self):
        logger.info("=" * 70)
        logger.info("Initialising AdvancedRAGPipeline…")
        logger.info("=" * 70)

        # Lazy initialisation of heavy components
        self._ingestion_engine  = DocumentIngestionEngine()
        self._vector_manager    = VectorStoreManager()
        self._reranker          = CrossEncoderReranker()
        self._synthesis_engine  = SynthesisEngine()
        self._evaluator         = RagasEvaluator()

        # These are built after ingestion
        self._hybrid_retriever: Optional[HybridRetriever] = None

        # Attempt to restore a previously persisted store
        if self._vector_manager.load_existing():
            self._rebuild_hybrid_retriever()

        logger.info("AdvancedRAGPipeline ready.")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, file_path: str) -> None:
        """
        Full ingestion pipeline for a single file:
          1. Load & chunk with structure-aware splitter.
          2. Embed & persist to Chroma.
          3. Rebuild hybrid retriever over entire corpus.
        """
        logger.info("─" * 60)
        logger.info("INGESTION: %s", file_path)
        logger.info("─" * 60)

        # Stage 1: chunk
        chunks = self._ingestion_engine.ingest(file_path)

        # Stage 2: embed + store
        self._vector_manager.add_documents(chunks)

        # Stage 3: rebuild hybrid retriever
        self._rebuild_hybrid_retriever()

        logger.info("Ingestion pipeline complete for '%s'.", file_path)

    def _rebuild_hybrid_retriever(self) -> None:
        """(Re-)constructs the HybridRetriever from current store contents."""
        all_docs = self._vector_manager.get_all_documents()
        vector_ret = self._vector_manager.get_vector_retriever(k=CANDIDATE_K)
        self._hybrid_retriever = HybridRetriever(
            vector_retriever=vector_ret,
            corpus_documents=all_docs,
            k=CANDIDATE_K,
        )

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """
        Full query pipeline:
          Stage 3 → Hybrid Retrieval (25 candidates)
          Stage 4 → Cross-Encoder Re-ranking (top 5)
          Stage 5 → Compression + LLM Synthesis

        Returns a dict with keys:
          question, answer, source_documents, contexts (raw strings)
        """
        if self._hybrid_retriever is None:
            raise RuntimeError(
                "No documents have been ingested yet. Call pipeline.ingest(file_path) first."
            )

        logger.info("─" * 60)
        logger.info("QUERY: %s", question)
        logger.info("─" * 60)

        # Stage 3: Hybrid retrieval
        candidates = self._hybrid_retriever.retrieve(question)

        # Stage 4: Re-ranking
        reranked = self._reranker.rerank(question, candidates)

        # Stage 5: Compression + Synthesis
        answer, compressed_docs = self._synthesis_engine.synthesize(question, reranked)

        logger.info("Query complete.")

        return {
            "question":         question,
            "answer":           answer,
            "source_documents": compressed_docs,
            "contexts":         [doc.page_content for doc in compressed_docs],
        }

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, samples: List[dict]) -> pd.DataFrame:
        """
        Evaluate one or more RAG samples using Ragas.

        Each sample dict must contain:
          question (str), answer (str), contexts (List[str]), ground_truth (str)
        """
        return self._evaluator.evaluate(samples)


# =============================================================================
# MAIN — Example: Ingest, Query, Evaluate
# =============================================================================

if __name__ == "__main__":

    # ── 0. Validate required environment variables ────────────────────────────
    missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
    if missing:
        logger.error(
            "Missing environment variable(s): %s\n"
            "Set them before running:\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "  export OPENAI_API_KEY='sk-...'",
            ", ".join(missing),
        )
        sys.exit(1)

    # ── 1. Initialise pipeline ────────────────────────────────────────────────
    pipeline = AdvancedRAGPipeline()

    # ── 2. Ingest a document ──────────────────────────────────────────────────
    #   Replace with your actual file path. Supports .pdf and .md files.
    DOCUMENT_PATH = "sample_research_paper.pdf"   # ← change to your file

    if not Path(DOCUMENT_PATH).exists():
        logger.warning(
            "Sample document '%s' not found.\n"
            "Creating a demo Markdown document for illustration purposes…",
            DOCUMENT_PATH,
        )
        # Create a small demo Markdown document so the script runs end-to-end
        DOCUMENT_PATH = "demo_document.md"
        demo_content = """\
# Quantum Computing in Financial Risk Modelling

## 1. Introduction

Quantum computing represents a paradigm shift in computational finance.
Traditional Monte Carlo simulations for Value-at-Risk (VaR) require O(N) samples
to achieve ε-accuracy. Quantum amplitude estimation reduces this to O(1/ε) queries,
delivering a quadratic speedup over classical methods.

## 2. Key Algorithms

### 2.1 Quantum Amplitude Estimation (QAE)

QAE was introduced by Brassard et al. (2002). Given a quantum operator A such that
A|0⟩ = √(1−a)|ψ₀⟩|0⟩ + √a|ψ₁⟩|1⟩, the algorithm estimates the probability a
with precision ε using M = O(1/ε) oracle calls.

### 2.2 Variational Quantum Eigensolver (VQE)

VQE approximates the ground state of a Hamiltonian H by minimising ⟨ψ(θ)|H|ψ(θ)⟩
over a parameterised ansatz |ψ(θ)⟩. Applications include portfolio optimisation
over Ising-model formulations.

## 3. Financial Metrics

The 99% VaR for a portfolio with normally distributed returns μ=0.05 and σ=0.12
is calculated as: VaR₀.₉₉ = μ − z₀.₉₉ × σ = 0.05 − 2.326 × 0.12 = −0.229 (22.9% loss).

### 3.1 CVaR (Conditional Value-at-Risk)

CVaR at the 99% confidence level equals −(μ − σ × φ(z₀.₉₉) / (1−0.99))
where φ is the standard normal PDF. For the above portfolio, CVaR₀.₉₉ ≈ 26.1%.

## 4. Conclusion

Quantum-enhanced Monte Carlo methods are projected to achieve practical advantage
on error-corrected hardware with ≥1000 logical qubits, expected circa 2028–2030
according to IBM Quantum roadmap projections.
"""
        with open(DOCUMENT_PATH, "w", encoding="utf-8") as fh:
            fh.write(demo_content)
        logger.info("Demo document written to '%s'.", DOCUMENT_PATH)

    pipeline.ingest(DOCUMENT_PATH)

    # ── 3. Run a query ────────────────────────────────────────────────────────
    USER_QUESTION = (
        "What is the mathematical formula for the 99% VaR and CVaR "
        "of the portfolio described in the document, and what quantum "
        "speedup does amplitude estimation provide over classical Monte Carlo?"
    )

    result = pipeline.query(USER_QUESTION)

    # ── 4. Print answer + source metadata ────────────────────────────────────
    print("\n" + "=" * 70)
    print("QUESTION:")
    print(f"  {result['question']}")
    print("\nANSWER:")
    print(result["answer"])
    print("\nSOURCE DOCUMENTS USED:")
    for i, doc in enumerate(result["source_documents"], start=1):
        meta = doc.metadata
        print(
            f"  [{i}] Source={meta.get('source','?')} | "
            f"Page={meta.get('page','?')} | "
            f"Section={meta.get('section_path','?')} | "
            f"Rerank={meta.get('rerank_score','?')}"
        )
        # Print a short excerpt of the compressed chunk
        excerpt = doc.page_content[:200].replace("\n", " ")
        print(f"       Excerpt: {excerpt}…")
    print("=" * 70)

    # ── 5. Ragas evaluation ───────────────────────────────────────────────────
    EVAL_SAMPLES = [
        {
            "question": USER_QUESTION,
            "answer":   result["answer"],
            "contexts": result["contexts"],
            "ground_truth": (
                "Quantum amplitude estimation achieves a quadratic speedup (O(1/ε) queries "
                "vs O(N) classical). The 99% VaR is calculated as μ − z₀.₉₉ × σ = "
                "0.05 − 2.326 × 0.12 = −0.229 (22.9% loss). CVaR₀.₉₉ ≈ 26.1%."
            ),
        },
        # Add more evaluation rows here as needed
        {
            "question": "What year does IBM project quantum practical advantage will be achieved?",
            "answer":   pipeline.query(
                "What year does IBM project quantum practical advantage will be achieved?"
            )["answer"],
            "contexts": pipeline.query(
                "What year does IBM project quantum practical advantage will be achieved?"
            )["contexts"],
            "ground_truth": (
                "IBM's Quantum roadmap projects practical advantage on error-corrected "
                "hardware with ≥1000 logical qubits around 2028–2030."
            ),
        },
    ]

    print("\nRunning Ragas Evaluation…")
    eval_df = pipeline.evaluate(EVAL_SAMPLES)

    print("\n" + "=" * 70)
    print("RAGAS EVALUATION SCORECARD")
    print("=" * 70)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(eval_df.to_string(index=False))
    print("\nAggregate Means:")
    numeric_cols = eval_df.select_dtypes(include="number").columns
    print(eval_df[numeric_cols].mean().to_string())
    print("=" * 70)
