"""
AGRIVISION AI - Complete Advanced RAG Pipeline Orchestrator
Coordinates Query Rewriting, Hybrid Retrieval, Cross-Encoder Reranking, Grounded Generation, and Observability Traces.
"""

import time
from typing import Dict, Any, Optional

from backend.rag.query_rewriter import query_rewriter
from backend.rag.query_router import query_router
from backend.rag.hybrid_retriever import hybrid_retriever
from backend.rag.reranker import reranker
from backend.rag.context_builder import context_builder
from backend.rag.answer_generator import answer_generator
from backend.rag.citation_manager import citation_manager
from backend.rag.langchain_rag import langchain_rag_pipeline


class AdvancedRAGPipeline:
    """Master Pipeline orchestrating the full Advanced RAG flow."""

    def process_query(
        self,
        user_query: str,
        custom_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end Advanced RAG pipeline with LangChain LCEL orchestrator.
        """
        return langchain_rag_pipeline.process_query(user_query, custom_filters=custom_filters)


# Global Singleton
advanced_rag_pipeline = AdvancedRAGPipeline()
