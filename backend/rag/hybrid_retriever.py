"""
AGRIVISION AI - Hybrid Search Engine
Combines BM25 keyword search, dense semantic vector search, and metadata matching with Reciprocal Rank Fusion.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from backend.config import RAG_HYBRID_ALPHA, RETRIEVAL_CANDIDATE_LIMIT
from backend.rag.vector_store import vector_store


class HybridRetriever:
    """Production Hybrid Retrieval combining BM25 keyword and dense semantic vector search."""

    def __init__(self, alpha: float = RAG_HYBRID_ALPHA):
        self.alpha = alpha
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_docs: List[Dict[str, Any]] = []
        self._init_bm25()

    def _init_bm25(self) -> None:
        """Initializes BM25 index over all documents in vector store."""
        docs = vector_store.documents
        if not docs:
            return
        self.corpus_docs = docs
        tokenized_corpus = [self._tokenize(doc.get("text", "")) for doc in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """Simple, fast agronomic word tokenizer."""
        return [w.lower() for w in text.split() if len(w) > 1]

    def retrieve(
        self,
        query: str,
        multi_queries: Optional[List[str]] = None,
        top_k: int = RETRIEVAL_CANDIDATE_LIMIT,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid multi-query search and ranks candidates by combined vector and BM25 scores.
        """
        # Ensure BM25 is synced with vector store
        if self.bm25 is None or len(self.corpus_docs) != len(vector_store.documents):
            self._init_bm25()

        all_queries = [query]
        if multi_queries:
            all_queries.extend(multi_queries)

        candidate_scores: Dict[str, Dict[str, Any]] = {}

        # 1. Vector Search for all query variations
        for q in all_queries:
            vec_results = vector_store.similarity_search(q, top_k=top_k, filters=metadata_filters)
            for rank, doc in enumerate(vec_results):
                doc_id = doc.get("id", f"doc_{rank}")
                if doc_id not in candidate_scores:
                    candidate_scores[doc_id] = {
                        "doc": doc,
                        "vector_score": float(doc.get("vector_score", 0.0)),
                        "bm25_score": 0.0,
                        "rrf_score": 1.0 / (60 + rank + 1)
                    }
                else:
                    candidate_scores[doc_id]["vector_score"] = max(
                        candidate_scores[doc_id]["vector_score"],
                        float(doc.get("vector_score", 0.0))
                    )
                    candidate_scores[doc_id]["rrf_score"] += (1.0 / (60 + rank + 1))

        # 2. BM25 Search
        if self.bm25 and self.corpus_docs:
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            max_bm25 = float(np.max(bm25_scores)) if len(bm25_scores) > 0 and np.max(bm25_scores) > 0 else 1.0

            # Rank by BM25
            bm25_ranked_indices = np.argsort(bm25_scores)[::-1][:top_k]
            for rank, idx in enumerate(bm25_ranked_indices):
                doc = self.corpus_docs[idx]
                doc_id = doc.get("id", f"doc_bm25_{idx}")
                normalized_bm25 = float(bm25_scores[idx]) / max_bm25

                if doc_id not in candidate_scores:
                    candidate_scores[doc_id] = {
                        "doc": doc,
                        "vector_score": 0.0,
                        "bm25_score": normalized_bm25,
                        "rrf_score": 1.0 / (60 + rank + 1)
                    }
                else:
                    candidate_scores[doc_id]["bm25_score"] = normalized_bm25
                    candidate_scores[doc_id]["rrf_score"] += (1.0 / (60 + rank + 1))

        # 3. Calculate Final Hybrid Blend & Metadata Bonuses
        final_candidates = []
        for doc_id, data in candidate_scores.items():
            doc = data["doc"]
            v_score = data["vector_score"]
            b_score = data["bm25_score"]
            rrf = data["rrf_score"]

            # Metadata matching bonus
            meta_bonus = 0.0
            if metadata_filters:
                if metadata_filters.get('crop') and metadata_filters['crop'].lower() in str(doc.get('crop', '')).lower():
                    meta_bonus += 0.15
                if metadata_filters.get('disease') and metadata_filters['disease'].lower() in str(doc.get('disease', '')).lower():
                    meta_bonus += 0.20

            # Weighted Hybrid Score
            hybrid_score = (self.alpha * v_score) + ((1.0 - self.alpha) * b_score) + meta_bonus + (rrf * 5.0)

            doc_result = dict(doc)
            doc_result["vector_score"] = round(v_score, 4)
            doc_result["bm25_score"] = round(b_score, 4)
            doc_result["hybrid_score"] = round(float(hybrid_score), 4)
            final_candidates.append(doc_result)

        # Sort by final hybrid score
        final_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return final_candidates[:top_k]


# Global Singleton
hybrid_retriever = HybridRetriever()
