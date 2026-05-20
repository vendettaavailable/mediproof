from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE_PATH = (
    Path(__file__).resolve().parents[1] / "knowledge_base" / "medical_knowledge.json"
)

INDEX_CACHE_PATH = Path(__file__).resolve().parent / "faiss_index.bin"
INDEX_META_PATH = Path(__file__).resolve().parent / "faiss_index.meta.json"

MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 3

MIN_SIMILARITY_SCORE = 0.35
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_CONTRADICTION_PATTERNS = (
    "does not cure", "cannot cure", "no cure",
    "no evidence", "not effective", "not supported",
    "scientifically unfounded", "misinformation",
)
DEFAULT_FALLBACK_EVIDENCE = {
    "content": (
        "Trusted medical guidance should come from reliable public health sources such as WHO, CDC, or NHS. "
        "If retrieved evidence is unavailable, treat the claim cautiously until it is verified."
    ),
    "source": "MediProof Fallback Guidance",
    "url": "",
    "score": 0.1,
}
TOPIC_KEYWORDS: Dict[str, tuple[str, ...]] = {
    "covid19": ("covid", "mask", "masks", "sars-cov-2", "coronavirus", "vaccine", "vaccines", "respiratory"),
    "general_health": ("health", "diet", "exercise", "vitamin", "vitamins", "supplement", "handwashing", "hand washing"),
    "mental_health": ("mental", "depression", "anxiety", "stress", "therapy", "psychiatric", "psychological"),
}


# ---------------------------------------------------------------------------
# Internal State
# ---------------------------------------------------------------------------

@dataclass
class _RAGStore:
    documents: List[Dict]
    index: Any
    model: Any
    doc_signature: str


_store: Optional[_RAGStore] = None
_lock = threading.Lock()


def _require_faiss():
    try:
        import faiss
        return faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FAISS is not installed. Install 'faiss-cpu' in the active Python environment."
        ) from exc


def _require_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install 'sentence-transformers' in the active Python environment."
        ) from exc


def _extract_sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        return []

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    parts = _SENTENCE_SPLIT_PATTERN.split(normalized)
    sentences = [part.strip() for part in parts if part and part.strip()]
    if sentences:
        return sentences

    return [normalized]


def _contains_contradiction(text: str) -> bool:
    """Check if text contains contradiction patterns."""
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in _CONTRADICTION_PATTERNS)


def _keyword_overlap_score(query: str, content: str) -> float:
    query_terms = set(re.findall(r"[a-zA-Z]{4,}", query.lower()))
    content_terms = set(re.findall(r"[a-zA-Z]{4,}", content.lower()))
    if not query_terms or not content_terms:
        return 0.0

    overlap = len(query_terms & content_terms)
    return overlap / max(len(query_terms), 1)


def _detect_query_topics(query: str) -> List[str]:
    query_lower = query.lower()
    matched_topics: List[str] = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            matched_topics.append(topic)

    return matched_topics


def _document_matches_query_topic(query: str, document: Dict[str, Any]) -> bool:
    matched_topics = _detect_query_topics(query)
    if not matched_topics:
        return True

    document_topic = str(document.get("topic", "")).lower()
    content = str(document.get("content", "")).lower()

    if document_topic in matched_topics:
        return True

    for topic in matched_topics:
        keywords = TOPIC_KEYWORDS.get(topic, ())
        if any(keyword in content for keyword in keywords):
            return True

    return False


def _filter_documents_for_query(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = [doc for doc in documents if _document_matches_query_topic(query, doc)]
    return filtered if filtered else documents


def _fallback_retrieve_evidence(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Simple keyword-based retrieval used when FAISS or sentence-transformers
    are not available. This keeps evidence non-empty for beginner-friendly demos.
    """
    documents = _filter_documents_for_query(query, _load_documents())
    if not documents:
        return [DEFAULT_FALLBACK_EVIDENCE]

    scored: List[Dict[str, Any]] = []
    for doc in documents:
        score = _keyword_overlap_score(query, doc.get("content", ""))
        if score <= 0:
            continue
        scored.append({
            "content": doc.get("content", ""),
            "source": doc.get("source", "Unknown Source"),
            "url": doc.get("url", ""),
            "score": round(score, 4),
        })

    scored.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    results = scored[: max(1, top_k)]
    if not results:
        print("Retrieved docs:", [DEFAULT_FALLBACK_EVIDENCE])
        return [DEFAULT_FALLBACK_EVIDENCE]
    print("Retrieved docs:", results)
    return results


# ---------------------------------------------------------------------------
# Load Documents
# ---------------------------------------------------------------------------

def _load_documents() -> List[Dict]:

    if not KNOWLEDGE_BASE_PATH.exists():
        logger.warning(
            "Knowledge base not found at %s. RAG retrieval will return no results.",
            KNOWLEDGE_BASE_PATH,
        )
        return []

    with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    docs: List[Dict] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        content = item.get("content", "").strip()

        if not content:
            continue

        docs.append({
            "topic": item.get("topic", ""),
            "content": content,
            "source": item.get("source", "Unknown Source"),
            "url": item.get("url", "")
        })

    logger.info("Loaded %d documents from knowledge base.", len(docs))

    return docs


# ---------------------------------------------------------------------------
# Build FAISS Index
# ---------------------------------------------------------------------------

def _documents_signature(documents: List[Dict]) -> str:
    payload = json.dumps(documents, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_index_metadata(documents: List[Dict], model_dim: int):
    metadata = {
        "model_name": MODEL_NAME,
        "document_count": len(documents),
        "document_signature": _documents_signature(documents),
        "embedding_dimension": model_dim,
    }
    INDEX_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _read_index_metadata() -> Optional[Dict[str, Any]]:
    if not INDEX_META_PATH.exists():
        return None
    try:
        return json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _build_index(documents: List[Dict], model):

    faiss = _require_faiss()
    import numpy as np

    if not documents:
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        _write_index_metadata(documents, dim)
        return index

    texts = [doc["content"] for doc in documents]

    logger.info("Encoding %d documents for FAISS index", len(texts))

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    logger.info("FAISS index built with %d vectors", index.ntotal)

    _write_index_metadata(documents, dim)

    return index


# ---------------------------------------------------------------------------
# Save / Load Index
# ---------------------------------------------------------------------------

def _save_index(index, path: Path):

    faiss = _require_faiss()

    try:
        faiss.write_index(index, str(path))
        logger.info("FAISS index saved to %s", path)
    except Exception as exc:
        logger.warning("Could not save FAISS index: %s", exc)


def _load_index(path: Path, documents: List[Dict]):

    faiss = _require_faiss()

    expected_signature = _documents_signature(documents)
    expected_docs = len(documents)
    expected_dim = 0

    metadata = _read_index_metadata()
    if metadata is not None:
        expected_dim = int(metadata.get("embedding_dimension", 0))

    if (
        metadata is None
        or metadata.get("model_name") != MODEL_NAME
        or metadata.get("document_count") != expected_docs
        or metadata.get("document_signature") != expected_signature
    ):
        return None

    try:

        if path.exists():

            index = faiss.read_index(str(path))

            if index.ntotal == expected_docs and (expected_dim == 0 or index.d == expected_dim):
                logger.info("FAISS index loaded from disk")
                return index

            logger.info("Index mismatch. Rebuilding index.")

    except Exception as exc:

        logger.warning("Could not load cached index: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Lazy RAG Store Initialisation
# ---------------------------------------------------------------------------

def _get_store() -> _RAGStore:

    global _store

    if _store is not None:
        return _store

    with _lock:

        if _store is not None:
            return _store

        SentenceTransformer = _require_sentence_transformer()

        logger.info("Loading embedding model: %s", MODEL_NAME)
        try:
            model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        except Exception:
            model = SentenceTransformer(MODEL_NAME)

        documents = _load_documents()
        doc_signature = _documents_signature(documents)

        index = _load_index(INDEX_CACHE_PATH, documents)

        if index is None:

            index = _build_index(documents, model)

            _save_index(index, INDEX_CACHE_PATH)

        _store = _RAGStore(
            documents=documents,
            index=index,
            model=model,
            doc_signature=doc_signature,
        )

        return _store


# ---------------------------------------------------------------------------
# Public Retrieval API
# ---------------------------------------------------------------------------

def retrieve_evidence(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    print("Query:", query)

    try:
        faiss = _require_faiss()
        import numpy as np
    except RuntimeError as exc:
        logger.error("RAG retrieval unavailable: %s", exc)
        fallback_results = _fallback_retrieve_evidence(query, top_k=top_k)
        logger.info("Using fallback retrieval, results=%d", len(fallback_results))
        return fallback_results

    if not isinstance(query, str) or not query.strip():
        return [DEFAULT_FALLBACK_EVIDENCE]

    try:
        store = _get_store()
    except RuntimeError as exc:
        logger.error("RAG retrieval unavailable: %s", exc)
        fallback_results = _fallback_retrieve_evidence(query, top_k=top_k)
        logger.info("Using fallback retrieval, results=%d", len(fallback_results))
        return fallback_results

    if store.index.ntotal == 0:
        logger.warning("FAISS index is empty")
        return _fallback_retrieve_evidence(query, top_k=top_k)

    safe_top_k = top_k if isinstance(top_k, int) and top_k > 0 else DEFAULT_TOP_K

    query_vector = store.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    query_vector = np.asarray(query_vector, dtype="float32")

    faiss.normalize_L2(query_vector)

    k = min(max(safe_top_k * 4, safe_top_k), store.index.ntotal)

    scores, indices = store.index.search(query_vector, k)

    candidates: List[Dict[str, Any]] = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        if float(score) < MIN_SIMILARITY_SCORE:
            continue

        doc = store.documents[idx]
        if not _document_matches_query_topic(query, doc):
            continue

        candidates.append({
            "topic": doc.get("topic", ""),
            "content": doc["content"],
            "source": doc["source"],
            "url": doc["url"],
            "score": float(score),
        })

    if not candidates:
        logger.info("Retrieved %d evidence passages", 0)
        return _fallback_retrieve_evidence(query, top_k=top_k)

    all_sentences: List[str] = []
    sentence_owner_idx: List[int] = []

    for candidate_idx, candidate in enumerate(candidates):
        sentences = _extract_sentences(candidate["content"])
        for sentence in sentences:
            all_sentences.append(sentence)
            sentence_owner_idx.append(candidate_idx)

    best_sentence_by_candidate: Dict[int, str] = {}

    if all_sentences:
        sentence_vectors = store.model.encode(
            all_sentences,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        sentence_vectors = np.asarray(sentence_vectors, dtype="float32")
        faiss.normalize_L2(sentence_vectors)

        sentence_scores = sentence_vectors @ query_vector[0]
        best_score_by_candidate: Dict[int, float] = {}

        for sentence_score, owner_idx, sentence in zip(sentence_scores, sentence_owner_idx, all_sentences):
            score_value = float(sentence_score)
            previous = best_score_by_candidate.get(owner_idx)
            if previous is None or score_value > previous:
                best_score_by_candidate[owner_idx] = score_value
                best_sentence_by_candidate[owner_idx] = sentence

    results = []
    seen_content_hashes: set = set()

    for candidate_idx, candidate in enumerate(candidates):
        content = best_sentence_by_candidate.get(candidate_idx) or candidate["content"]
        content_hash = hashlib.md5(content.lower().encode()).hexdigest()

        if content_hash in seen_content_hashes:
            continue
        seen_content_hashes.add(content_hash)

        results.append({
            "content": content,
            "source": candidate["source"],
            "url": candidate["url"],
            "score": round(candidate["score"], 4),
        })

        if len(results) >= 3:
            break

    results.sort(key=lambda item: item["score"], reverse=True)

    logger.info("Retrieved %d evidence passages", len(results))
    print("Retrieved docs:", results)
    if not results:
        return _fallback_retrieve_evidence(query, top_k=top_k)
    return results


# ---------------------------------------------------------------------------
# Cache Reset
# ---------------------------------------------------------------------------

def invalidate_cache():

    global _store

    with _lock:

        _store = None

        if INDEX_CACHE_PATH.exists():
            INDEX_CACHE_PATH.unlink()

        if INDEX_META_PATH.exists():
            INDEX_META_PATH.unlink()

    logger.info("RAG cache invalidated")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        print("Loading knowledge base...")
        documents = _load_documents()
        print(f"Total documents: {len(documents)}")

        print("Generating embeddings...")
        SentenceTransformer = _require_sentence_transformer()
        model = SentenceTransformer(MODEL_NAME)

        print("Building FAISS index...")
        index = _build_index(documents, model)
        _save_index(index, INDEX_CACHE_PATH)

        print("Index saved successfully.")
    except Exception as exc:
        print(f"Failed to rebuild FAISS index: {exc}")
        raise
