"""
AGRIVISION AI - LangChain Advanced RAG Orchestration Engine
Integrates LangChain LCEL (LangChain Expression Language), custom BaseRetriever,
ChatGoogleGenerativeAI (Google Gemini), ChatGroq, and evidence citation managers.
"""

import time
import os
import re
from typing import Dict, Any, List, Optional
from pydantic import Field

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    DEFAULT_LLM_PROVIDER,
    RETRIEVAL_CANDIDATE_LIMIT,
    RERANKED_EVIDENCE_LIMIT
)
from backend.rag.query_rewriter import query_rewriter
from backend.rag.query_router import query_router
from backend.rag.hybrid_retriever import hybrid_retriever
from backend.rag.reranker import reranker
from backend.rag.context_builder import context_builder
from backend.rag.citation_manager import citation_manager
from backend.rag.image_retriever import image_retriever


AGRI_SYSTEM_PROMPT = """You are AgriMind, an expert AI Agricultural Advisor talking directly to a farmer.
Your goal is to provide clear, friendly, action-oriented, point-by-point advice that is easy to read.

CRITICAL FORMATTING & STRUCTURE RULES:
1. CROP NAME & WHY MUST ALWAYS COME FIRST:
   The very first section of your answer MUST clearly state the specific Crop Name(s) and WHY:
   - For crop selection / low budget / recommendation questions:
     ## 🌱 Recommended Crops & Why
     - **[Crop Name]**: Why it is suitable (e.g. low seed/fertilizer cost, fast 45-60 day harvest cycle, nitrogen-fixing, high market demand).
   - For disease / pest / nutrient problem questions:
     ## 🌱 Target Crop & Diagnosis
     - **[Crop Name] — [Identified Issue]**: Why this occurs (key leaf symptoms and causes).

2. POINT-BY-POINT FORMAT ONLY: Always write in clean, separate bullet points (- ). NEVER generate markdown tables (| Col 1 | Col 2 |), pipe characters, ASCII boxes, or long text paragraphs.
3. ONE ACTION PER POINT: Every bullet point must communicate ONE clear, actionable step in a short sentence.
4. STRUCTURED HEADINGS TO FOLLOW AFTER CROP & WHY:
   ## 🩺 What to Do Now
   - Immediate step-by-step actions to take in the field today.
   ## 💊 Recommended Treatment & Medicine (or Organic / Soil Inputs)
   - Point-by-point recommended medicines/fertilizers with label dosages supported by retrieved evidence.
   ## 🛡️ Prevention Protocol
   - Point-by-point preventive cultural practices, spacing, and crop rotation.
   ## ⚠️ Important Precautions
   - Safety gear, product label compliance, and pre-harvest intervals.
5. EVIDENCE GROUNDED: Strictly use retrieved agriculture context. Never hallucinate unregistered chemicals or dosages. If dosage is not in evidence, state: "Follow the approved product label."
6. SIMPLE FARMER-FRIENDLY LANGUAGE: Use simple words. Instead of "foliar pathogen proliferation", say "fungal disease spreads faster when leaves stay wet".
7. MULTILINGUAL: Respond in the farmer's language (English, Telugu, Hindi, or natural mix).
"""


class AgriHybridRetriever(BaseRetriever):
    """
    LangChain-compliant Custom Retriever wrapping BM25 and Dense Vector Hybrid Search.
    """
    top_k: int = RETRIEVAL_CANDIDATE_LIMIT
    metadata_filters: Optional[Dict[str, Any]] = None

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Synchronous retrieval of agricultural document chunks."""
        raw_candidates = hybrid_retriever.retrieve(
            query=query,
            top_k=self.top_k,
            metadata_filters=self.metadata_filters
        )
        docs = []
        for c in raw_candidates:
            doc = Document(
                page_content=c.get("text", ""),
                metadata={
                    "id": c.get("id", ""),
                    "source": c.get("source", "Agronomy Knowledge Base"),
                    "source_type": c.get("source_type", "Document"),
                    "document_name": c.get("document_name", ""),
                    "page": c.get("page", 1),
                    "crop": c.get("crop", "General Agriculture"),
                    "category": c.get("category", "General"),
                    "vector_score": c.get("vector_score", 0.0),
                    "bm25_score": c.get("bm25_score", 0.0),
                    "hybrid_score": c.get("hybrid_score", 0.0)
                }
            )
            docs.append(doc)
        return docs


class LangChainAdvancedRAG:
    """Master LangChain Advanced RAG Engine."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or DEFAULT_LLM_PROVIDER
        self.llm = self._init_chat_model()
        self.retriever = AgriHybridRetriever()

    def _init_chat_model(self):
        """Initializes LangChain Chat Model with dynamic fallback."""
        # 1. Google Gemini via langchain_google_genai
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY
        if gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                gemini_models = [
                    GEMINI_MODEL,
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                    "gemini-2.0-flash",
                    "gemini-2.5-flash"
                ]
                for g_model in dict.fromkeys(gemini_models):
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model=g_model,
                            google_api_key=gemini_key,
                            temperature=0.1,
                            max_output_tokens=1500
                        )
                        print(f"[LangChainRAG] Initialized ChatGoogleGenerativeAI with model: {g_model}")
                        return llm
                    except Exception as e:
                        print(f"[LangChainRAG] Failed initializing Gemini model '{g_model}': {e}")
            except Exception as e:
                print(f"[LangChainRAG] langchain_google_genai import/init error: {e}")

        # 2. Groq via langchain_groq
        groq_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
        if groq_key:
            try:
                from langchain_groq import ChatGroq
                groq_models = [
                    GROQ_MODEL,
                    "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b",
                    "qwen/qwen3.6-27b",
                    "groq/compound",
                    "groq/compound-mini",
                    "llama-3.3-70b-versatile"
                ]
                for q_model in groq_models:
                    if not q_model:
                        continue
                    try:
                        llm = ChatGroq(
                            model_name=q_model,
                            groq_api_key=groq_key,
                            temperature=0.2,
                            max_tokens=1500
                        )
                        print(f"[LangChainRAG] Initialized ChatGroq with model: {q_model}")
                        return llm
                    except Exception as e:
                        print(f"[LangChainRAG] Failed initializing Groq model '{q_model}': {e}")
            except Exception as e:
                print(f"[LangChainRAG] langchain_groq import/init error: {e}")

        print("[LangChainRAG] Notice: Running in local offline grounding mode.")
        return None

    def process_query(
        self,
        user_query: str,
        custom_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end LangChain Advanced RAG Pipeline with full LCEL integration
        and observability telemetry.
        """
        start_time = time.time()
        trace: Dict[str, Any] = {
            "stages": [],
            "timings": {},
            "framework": "LangChain LCEL"
        }

        # Step 1: Query Analysis & Transformation
        t0 = time.time()
        transformation = query_rewriter.process(user_query)
        rewritten_q = transformation["rewritten_query"]
        multi_queries = transformation["multi_queries"]
        metadata = transformation["metadata"]
        t_rewrite = round((time.time() - t0) * 1000, 2)
        trace["timings"]["query_rewriting_ms"] = t_rewrite
        trace["stages"].append({
            "stage": "Query Transformation",
            "original_query": user_query,
            "rewritten_query": rewritten_q,
            "multi_queries": multi_queries,
            "detected_metadata": metadata
        })

        # Step 2: Query Routing
        t0 = time.time()
        routing = query_router.route(rewritten_q, metadata)
        t_route = round((time.time() - t0) * 1000, 2)
        trace["timings"]["query_routing_ms"] = t_route
        trace["stages"].append({
            "stage": "Query Routing",
            "category": routing["category"],
            "pipeline": routing["pipeline"]
        })

        # Step 3: LangChain Hybrid Retrieval
        t0 = time.time()
        combined_filters = {
            **(custom_filters or {}),
            **({'crop': metadata['crop']} if metadata.get('crop') else {})
        }
        self.retriever.metadata_filters = combined_filters

        # Retrieve for all query expansions
        all_candidate_dicts: List[Dict[str, Any]] = []
        seen_ids = set()

        all_queries = [rewritten_q] + [q for q in multi_queries if q != rewritten_q]
        for q_variant in all_queries:
            lc_docs = self.retriever._get_relevant_documents(q_variant)
            for d in lc_docs:
                doc_id = d.metadata.get("id", "")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_candidate_dicts.append({
                        "id": doc_id,
                        "text": d.page_content,
                        "source": d.metadata.get("source", ""),
                        "source_type": d.metadata.get("source_type", ""),
                        "document_name": d.metadata.get("document_name", ""),
                        "page": d.metadata.get("page", 1),
                        "crop": d.metadata.get("crop", "General Agriculture"),
                        "category": d.metadata.get("category", "General"),
                        "vector_score": d.metadata.get("vector_score", 0.0),
                        "bm25_score": d.metadata.get("bm25_score", 0.0),
                        "hybrid_score": d.metadata.get("hybrid_score", 0.0)
                    })

        t_retrieval = round((time.time() - t0) * 1000, 2)
        trace["timings"]["hybrid_retrieval_ms"] = t_retrieval
        trace["stages"].append({
            "stage": "Hybrid Retrieval (LangChain)",
            "candidates_count": len(all_candidate_dicts),
            "top_candidates": [
                {
                    "source": c.get("source"),
                    "vector_score": c.get("vector_score"),
                    "bm25_score": c.get("bm25_score"),
                    "hybrid_score": c.get("hybrid_score")
                }
                for c in all_candidate_dicts[:5]
            ]
        })

        # Step 4: Cross-Encoder Contextual Reranking
        t0 = time.time()
        reranked_evidence = reranker.rerank(
            query=rewritten_q,
            candidates=all_candidate_dicts,
            metadata_focus=metadata,
            top_n=RERANKED_EVIDENCE_LIMIT
        )
        t_rerank = round((time.time() - t0) * 1000, 2)
        trace["timings"]["reranking_ms"] = t_rerank
        trace["stages"].append({
            "stage": "Reranking",
            "evidence_count": len(reranked_evidence),
            "reranked_scores": [
                {
                    "source": e.get("source"),
                    "reranker_score": e.get("rerank_score"),
                    "relevance_pct": e.get("relevance_pct")
                }
                for e in reranked_evidence
            ]
        })

        # Step 5: Context Assembly
        t0 = time.time()
        context_data = context_builder.assemble_context(reranked_evidence)
        t_context = round((time.time() - t0) * 1000, 2)
        trace["timings"]["context_compression_ms"] = t_context

        # Step 6: Grounded Generation (LangChain LCEL Chain)
        t0 = time.time()
        formatted_context = context_data.get("formatted_context", "")
        evidence_items = context_data.get("evidence_items", [])

        answer_text = None
        if self.llm and evidence_items:
            try:
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", AGRI_SYSTEM_PROMPT),
                    ("human", (
                        "Farmer Question: {original_query}\n"
                        "Agronomic Interpreted Formulation: {rewritten_query}\n\n"
                        "RETRIEVED EVIDENCE FROM AGRICULTURE KNOWLEDGE BASE:\n{context}\n\n"
                        "Provide a structured, authoritative, actionable response with [1], [2] citations."
                    ))
                ])

                lcel_chain = prompt_template | self.llm | StrOutputParser()
                answer_text = lcel_chain.invoke({
                    "original_query": user_query,
                    "rewritten_query": rewritten_q,
                    "context": formatted_context
                })
            except Exception as e:
                print(f"[LangChainRAG] LCEL Chain invocation error: {e}")

        # Fallback to answer generator if LCEL model fails
        if not answer_text:
            from backend.rag.answer_generator import answer_generator
            gen_res = answer_generator.generate_answer(
                original_query=user_query,
                rewritten_query=rewritten_q,
                context_data=context_data,
                metadata=metadata
            )
            answer_text = gen_res["answer"]

        t_gen = round((time.time() - t0) * 1000, 2)
        trace["timings"]["generation_ms"] = t_gen

        # Step 7: Visual Disease & Pest Image Retrieval
        t0 = time.time()
        visual_images = image_retriever.search_images(
            query=user_query,
            crop=metadata.get("crop"),
            disease=metadata.get("disease"),
            pest=metadata.get("pest"),
            top_k=4
        )
        t_img = round((time.time() - t0) * 1000, 2)
        trace["timings"]["image_retrieval_ms"] = t_img
        trace["stages"].append({
            "stage": "Visual Image Retrieval",
            "images_retrieved": len(visual_images),
            "image_labels": [img.get("label") for img in visual_images]
        })

        # Step 8: Citation Validation & Grounding Metrics
        validation = citation_manager.validate_grounding(answer_text, evidence_items)
        total_time_ms = round((time.time() - start_time) * 1000, 2)
        trace["timings"]["total_latency_ms"] = total_time_ms

        return {
            "answer": answer_text,
            "category": routing["category"],
            "evidence_items": evidence_items,
            "visual_evidence": visual_images,
            "is_grounded": validation["is_grounded"],
            "grounding_level": validation["grounding_level"],
            "grounding_score": validation["grounding_score"],
            "original_query": user_query,
            "interpreted_query": rewritten_q,
            "query_transformation": transformation,
            "latency_ms": total_time_ms,
            "trace": trace
        }


# Global Singleton
langchain_rag_pipeline = LangChainAdvancedRAG()
