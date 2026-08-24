"""
AGRIVISION AI - Embedding Engine
Multi-provider semantic embeddings supporting cloud APIs (Gemini/OpenAI) and local dense semantic vectorizers.
"""

from typing import List, Union
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from backend.config import GEMINI_API_KEY, OPENAI_API_KEY, EMBEDDING_PROVIDER


class SemanticEmbeddingEngine:
    """Robust embedding manager supporting cloud LLM embeddings and local semantic vectors."""

    def __init__(self, provider: str = EMBEDDING_PROVIDER):
        self.provider = provider
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=2500,
            sublinear_tf=True,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b"
        )
        self.is_fitted = False
        self.dimension = 384

    def fit_local_vectorizer(self, corpus: List[str]) -> None:
        """Fits the local semantic vectorizer on the agricultural corpus."""
        if not corpus:
            return
        valid_texts = [t for t in corpus if t and t.strip()]
        if len(valid_texts) == 0:
            return
        self.vectorizer.fit(valid_texts)
        self.is_fitted = True

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Computes dense normalized embeddings for a list of document chunks."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Check cloud embedding if explicitly configured and API key present
        if self.provider == "gemini" and GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                embeddings = []
                for text in texts:
                    result = genai.embed_content(
                        model="models/embedding-001",
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(result['embedding'])
                arr = np.array(embeddings, dtype=np.float32)
                # L2 normalize
                norm = np.linalg.norm(arr, axis=1, keepdims=True)
                norm[norm == 0] = 1.0
                return arr / norm
            except Exception as e:
                print(f"[EmbeddingEngine] Gemini embedding failed, falling back to local: {e}")

        # Local semantic dense projection
        if not self.is_fitted:
            self.fit_local_vectorizer(texts)

        sparse_matrix = self.vectorizer.transform(texts)
        dense = sparse_matrix.toarray().astype(np.float32)

        # Pad or trim to target dimension
        if dense.shape[1] < self.dimension:
            padded = np.zeros((dense.shape[0], self.dimension), dtype=np.float32)
            padded[:, :dense.shape[1]] = dense
            dense = padded
        elif dense.shape[1] > self.dimension:
            dense = dense[:, :self.dimension]

        # L2 normalize
        norm = np.linalg.norm(dense, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return dense / norm

    def embed_query(self, query: str) -> np.ndarray:
        """Computes a dense normalized embedding for a search query."""
        res = self.embed_documents([query])
        return res[0]


# Global Singleton
embedding_engine = SemanticEmbeddingEngine()
