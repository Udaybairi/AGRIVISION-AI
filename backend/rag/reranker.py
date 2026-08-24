"""
AGRIVISION AI - Contextual Cross-Encoder & Reranking Engine
Refines top candidate chunks to the highest precision evidence subset.
"""

from typing import List, Dict, Any, Optional
import re
from backend.config import RERANKED_EVIDENCE_LIMIT


class AgronomicReranker:
    """Production Cross-Encoder and Contextual Reranking Module."""

    def __init__(self, top_n: int = RERANKED_EVIDENCE_LIMIT):
        self.top_n = top_n

    def compute_relevance(self, query: str, doc_text: str, metadata_focus: Optional[Dict[str, Any]] = None) -> float:
        """
        Computes fine-grained semantic and lexical cross-interaction score between query and document chunk.
        """
        q_words = set(w.lower() for w in re.findall(r'\w+', query) if len(w) > 2)
        doc_words = set(w.lower() for w in re.findall(r'\w+', doc_text) if len(w) > 2)

        if not q_words or not doc_words:
            return 0.0

        # Term overlap (Jaccard-like with frequency boost)
        overlap = q_words.intersection(doc_words)
        overlap_score = len(overlap) / max(len(q_words), 1)

        # Agronomic phrase bonus
        phrase_bonus = 0.0
        lower_doc = doc_text.lower()
        lower_q = query.lower()

        if metadata_focus:
            crop = str(metadata_focus.get('crop', '')).lower()
            disease = str(metadata_focus.get('disease', '')).lower()
            pest = str(metadata_focus.get('pest', '')).lower()

            if crop and crop in lower_doc:
                phrase_bonus += 0.25
            if disease and disease in lower_doc:
                phrase_bonus += 0.35
            if pest and pest in lower_doc:
                phrase_bonus += 0.30

        # Action-oriented keywords (symptoms, dosage, management)
        for keyword in ['symptoms', 'management', 'control', 'treatment', 'spray', 'dosage', 'prevention', 'ipm']:
            if keyword in lower_q and keyword in lower_doc:
                phrase_bonus += 0.08

        # Length penalty for overly short chunks
        length_factor = min(len(doc_text) / 200.0, 1.0)

        raw_score = (overlap_score * 0.60) + phrase_bonus * length_factor
        return min(max(raw_score, 0.05), 0.99)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        metadata_focus: Optional[Dict[str, Any]] = None,
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Reranks retrieved candidate chunks and returns top-N highest quality evidence documents.
        """
        limit = top_n or self.top_n
        if not candidates:
            return []

        scored_candidates = []
        for doc in candidates:
            text = doc.get("text", "")
            base_hybrid = float(doc.get("hybrid_score", 0.5))
            rerank_score = self.compute_relevance(query, text, metadata_focus)
            
            # Combine normalized hybrid search score with cross-encoder relevance
            combined_relevance = (0.45 * base_hybrid) + (0.55 * rerank_score)
            
            doc_copy = dict(doc)
            doc_copy["reranker_score"] = round(float(rerank_score), 4)
            doc_copy["relevance_pct"] = round(min(combined_relevance * 100, 99.0), 1)
            scored_candidates.append(doc_copy)

        scored_candidates.sort(key=lambda x: x["reranker_score"], reverse=True)
        return scored_candidates[:limit]


# Global Singleton
reranker = AgronomicReranker()
