import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
from backend.rag.query_rewriter import query_rewriter
from backend.rag.query_router import query_router
from backend.rag.hybrid_retriever import hybrid_retriever
from backend.rag.reranker import reranker
from backend.rag.citation_manager import citation_manager
from backend.rag.pipeline import advanced_rag_pipeline


def test_query_rewriting_benchmark():
    """Verifies that colloquial farmer questions are transformed into agronomic queries."""
    # Test colloquial tomato query
    res1 = query_rewriter.process("tomato leaf black what medicine")
    assert "black" in res1["rewritten_query"].lower()
    assert "tomato" in res1["rewritten_query"].lower()
    assert len(res1["multi_queries"]) >= 2
    assert res1["metadata"]["crop"] == "Tomato"

    # Test rice yellow leaves query
    res2 = query_rewriter.process("rice yellow leaves fertilizer?")
    assert "rice" in res2["rewritten_query"].lower()
    assert "yellow" in res2["rewritten_query"].lower() or "nutrient" in res2["rewritten_query"].lower()


def test_query_routing():
    """Verifies intent categorization."""
    meta = {"crop": "Tomato", "disease": "Early Blight"}
    res = query_router.route("how to treat early blight on tomato", meta)
    assert res["category"] in ["Disease Identification", "Disease Management"]


def test_hybrid_retrieval_and_reranking():
    """Verifies hybrid retrieval candidate gathering and cross-encoder reranking."""
    query = "What causes concentric black rings on tomato leaves?"
    candidates = hybrid_retriever.retrieve(query, top_k=15)
    assert len(candidates) > 0

    reranked = reranker.rerank(query, candidates, top_n=5)
    assert len(reranked) <= 5
    assert "reranker_score" in reranked[0]
    assert "relevance_pct" in reranked[0]


def test_full_rag_pipeline_trace():
    """Verifies end-to-end execution and observability generation."""
    output = advanced_rag_pipeline.process_query("cotton pest spray")
    assert output is not None
    assert "original_query" in output
    assert "interpreted_query" in output
    assert "answer" in output
    assert len(output["evidence_items"]) > 0
    assert "trace" in output
    assert "timings" in output["trace"]
