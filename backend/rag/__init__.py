"""
AGRIVISION AI - Advanced Retrieval-Augmented Generation Subsystem
"""

from backend.rag.ingestion import data_ingestion, AgricultureDataIngestion
from backend.rag.embeddings import embedding_engine, SemanticEmbeddingEngine
from backend.rag.vector_store import vector_store, VectorStore
from backend.rag.hybrid_retriever import hybrid_retriever, HybridRetriever
from backend.rag.query_rewriter import query_rewriter, QueryRewriter
from backend.rag.query_router import query_router, QueryRouter
from backend.rag.reranker import reranker, AgronomicReranker
from backend.rag.context_builder import context_builder, ContextBuilder
from backend.rag.citation_manager import citation_manager, CitationManager
from backend.rag.answer_generator import answer_generator, AnswerGenerator
from backend.rag.pipeline import advanced_rag_pipeline, AdvancedRAGPipeline

__all__ = [
    "data_ingestion", "AgricultureDataIngestion",
    "embedding_engine", "SemanticEmbeddingEngine",
    "vector_store", "VectorStore",
    "hybrid_retriever", "HybridRetriever",
    "query_rewriter", "QueryRewriter",
    "query_router", "QueryRouter",
    "reranker", "AgronomicReranker",
    "context_builder", "ContextBuilder",
    "citation_manager", "CitationManager",
    "answer_generator", "AnswerGenerator",
    "advanced_rag_pipeline", "AdvancedRAGPipeline"
]
