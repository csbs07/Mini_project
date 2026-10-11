"""Hybrid Indexing (BM25 + Dense SentenceTransformer + Reciprocal Rank Fusion k=60)."""
import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME, RRF_K

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy singleton for sentence-transformers embedding model."""
    global _embedding_model
    if _embedding_model is None:
        # Load general text embedding model (all-MiniLM-L6-v2)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


class SessionChunkIndex:
    """
    Session-level index storing chunks from all ingested papers in the session pool.
    Performs hybrid BM25 + dense retrieval fused via Reciprocal Rank Fusion (RRF k=60).
    """

    def __init__(self):
        self.chunks: list[dict[str, Any]] = []
        self.chunk_embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: list[list[str]] = []

    def add_chunks(self, new_chunks: list[dict[str, Any]]) -> None:
        """Add new paper chunks to the session pool and recompute index."""
        if not new_chunks:
            return

        # Deduplicate chunks by chunk_id
        existing_ids = {c["chunk_id"] for c in self.chunks}
        filtered = [c for c in new_chunks if c["chunk_id"] not in existing_ids]
        if not filtered:
            return

        self.chunks.extend(filtered)

        # Compute embeddings for new chunks
        model = get_embedding_model()
        new_texts = [c["text"] for c in filtered]
        new_embs = model.encode(new_texts, convert_to_numpy=True, show_progress_bar=False)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(new_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        new_embs = new_embs / norms

        if self.chunk_embeddings is None:
            self.chunk_embeddings = new_embs
        else:
            self.chunk_embeddings = np.vstack([self.chunk_embeddings, new_embs])

        # Update BM25
        self.tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        paper_ids: list[str] | None = None,
        k: int = 6,
        include_references: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Hybrid retrieval over chunks using BM25 and dense embeddings with RRF (k=60).
        Excludes reference sections by default.
        Returns top k chunks (capped at 6).
        """
        if not self.chunks or self.bm25 is None or self.chunk_embeddings is None:
            return []

        k = max(1, min(k, 6))

        # Filter candidate indices based on paper_ids and references exclusion
        candidate_indices = []
        clean_target_papers = set(paper_ids) if paper_ids else None

        for idx, chunk in enumerate(self.chunks):
            # Exclude references by default
            if not include_references and chunk.get("is_references", False):
                continue
            # Filter by paper_ids or paper_labels if specified
            if clean_target_papers:
                pid = chunk.get("paper_id")
                plabel = chunk.get("paper_label")
                if pid not in clean_target_papers and plabel not in clean_target_papers:
                    continue
            candidate_indices.append(idx)

        if not candidate_indices:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # 1. BM25 scoring
        bm25_all_scores = self.bm25.get_scores(query_tokens)
        bm25_candidate_scores = [(idx, bm25_all_scores[idx]) for idx in candidate_indices]
        bm25_candidate_scores.sort(key=lambda x: x[1], reverse=True)
        bm25_ranks = {item[0]: rank for rank, item in enumerate(bm25_candidate_scores)}

        # 2. Dense scoring
        model = get_embedding_model()
        q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        candidate_embs = self.chunk_embeddings[candidate_indices]
        dense_scores = np.dot(candidate_embs, q_emb)
        dense_candidate_scores = [
            (candidate_indices[i], float(dense_scores[i])) for i in range(len(candidate_indices))
        ]
        dense_candidate_scores.sort(key=lambda x: x[1], reverse=True)
        dense_ranks = {item[0]: rank for rank, item in enumerate(dense_candidate_scores)}

        # 3. Reciprocal Rank Fusion (RRF k=60)
        rrf_scores = []
        for idx in candidate_indices:
            r_bm25 = bm25_ranks.get(idx, len(candidate_indices))
            r_dense = dense_ranks.get(idx, len(candidate_indices))
            rrf_score = (1.0 / (RRF_K + r_bm25)) + (1.0 / (RRF_K + r_dense))
            rrf_scores.append((idx, rrf_score))

        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_items = rrf_scores[:k]

        results = []
        for idx, score in top_k_items:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "paper_label": chunk["paper_label"],
                "paper_id": chunk["paper_id"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "section": chunk["section_guess"],
                "text": chunk["text"],
                "score": round(score, 6),
            })

        return results
