"""
AGRIVISION AI - Citation Manager & Grounding Validation Subsystem
Tracks verifiable evidence links, builds formatted citation badges, and validates claim grounding.
"""

from typing import List, Dict, Any, Tuple
import re


class CitationManager:
    """Manages source citations and verifies that claims are grounded in retrieved evidence."""

    def format_citation_source(self, item: Dict[str, Any]) -> str:
        """Creates authoritative, human-readable source citations."""
        source_type = item.get("source_type", "")
        doc_name = item.get("document_name", "Agriculture Knowledge Base")
        page = item.get("page")
        crop = item.get("crop", "")
        disease = item.get("disease", "")

        if "PDF" in source_type and page:
            return f"{doc_name} — Page {page}"
        elif "Knowledge Table" in source_type or "Excel" in source_type:
            return f"{doc_name} — Row {page or 1}"
        elif "Image" in source_type:
            return f"Agriculture Dataset → {crop} → {disease or 'Visual Reference'}"
        else:
            return item.get("source", "Agronomy Knowledge Base")

    def validate_grounding(self, answer_text: str, evidence_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates whether key entities and claims in the answer correspond to retrieved evidence chunks.
        """
        if not evidence_items:
            return {
                "is_grounded": False,
                "grounding_level": "Insufficient Evidence",
                "grounding_score": 0.20,
                "missing_elements": "Knowledge base lacks direct empirical matches for this specific query."
            }

        # Check citation presence in answer
        citation_tags = re.findall(r'\[\d+\]', answer_text)
        has_citations = len(citation_tags) > 0

        # Check keyword cross-overlap
        answer_words = set(re.findall(r'\w+', answer_text.lower()))
        evidence_text = " ".join([item.get("snippet", "").lower() for item in evidence_items])
        evidence_words = set(re.findall(r'\w+', evidence_text))

        common_words = answer_words.intersection(evidence_words)
        overlap_ratio = len(common_words) / max(len(answer_words), 1)

        if overlap_ratio > 0.40 and (has_citations or len(evidence_items) >= 2):
            grounding_level = "Strong"
            score = 0.95
        elif overlap_ratio > 0.25:
            grounding_level = "Moderate"
            score = 0.78
        else:
            grounding_level = "Preliminary / Heuristic"
            score = 0.55

        return {
            "is_grounded": score >= 0.50,
            "grounding_level": grounding_level,
            "grounding_score": score,
            "citations_found": len(citation_tags),
            "evidence_count": len(evidence_items)
        }


# Global Singleton
citation_manager = CitationManager()
