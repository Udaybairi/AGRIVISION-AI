"""
AGRIVISION AI - Contextual Compression & Assembly Module
Compresses retrieved evidence chunks, filters extraneous noise, and formats structured prompt context for LLM generation.
"""

from typing import List, Dict, Any


class ContextBuilder:
    """Context Compressor and Evidence Assembler."""

    def compress_chunk(self, text: str) -> str:
        """Strips conversational noise and extracts actionable agronomic sentences."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        compressed_lines = []

        for line in lines:
            # Filter out non-informative headers or duplicate table artifacts
            if any(ignore in line.lower() for ignore in ['page intentionally left blank', 'table of contents', 'copyright']):
                continue
            compressed_lines.append(line)

        return " ".join(compressed_lines)

    def assemble_context(self, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assembles compressed evidence blocks and formatted prompt context for the LLM.
        """
        if not evidence_chunks:
            return {
                "formatted_context": "No verified agricultural documents retrieved for this query.",
                "evidence_items": []
            }

        context_blocks = []
        evidence_items = []

        for idx, doc in enumerate(evidence_chunks, start=1):
            raw_text = doc.get("text", "")
            compressed = self.compress_chunk(raw_text)
            source_name = doc.get("source", f"Source {idx}")
            relevance = doc.get("relevance_pct", 85.0)

            context_blocks.append(
                f"[Source {idx}]: {source_name}\n"
                f"Content: {compressed}\n"
            )

            evidence_items.append({
                "citation_id": idx,
                "citation_tag": f"[{idx}]",
                "source": source_name,
                "document_name": doc.get("document_name", "Agriculture Knowledge Base"),
                "page": doc.get("page"),
                "file_path": doc.get("file_path"),
                "crop": doc.get("crop"),
                "category": doc.get("category"),
                "snippet": compressed[:220] + ("..." if len(compressed) > 220 else ""),
                "relevance_pct": relevance
            })

        formatted_text = "\n---\n".join(context_blocks)
        return {
            "formatted_context": formatted_text,
            "evidence_items": evidence_items
        }


# Global Singleton
context_builder = ContextBuilder()
