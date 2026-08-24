"""
AGRIVISION AI - Vector Store & Metadata Document Store
Persistent vector storage with cosine similarity search and metadata filtering.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from backend.config import VECTOR_STORE_DIR
from backend.rag.embeddings import embedding_engine


class VectorStore:
    """Vector database and document repository with metadata filtering."""

    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = store_dir or VECTOR_STORE_DIR
        self.documents: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None
        self.doc_file = self.store_dir / "documents.json"
        self.vec_file = self.store_dir / "vectors.npy"
        self.load_index()

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Indexes new document chunks and computes/updates dense embeddings."""
        if not docs:
            return

        texts = [doc.get("text", "") for doc in docs]
        embedding_engine.fit_local_vectorizer(texts)
        new_vectors = embedding_engine.embed_documents(texts)

        self.documents = docs
        self.vectors = new_vectors
        self.save_index()
        print(f"[VectorStore] Indexed {len(docs)} documents. Vector matrix shape: {self.vectors.shape}")

    def similarity_search(
        self,
        query: str,
        top_k: int = 15,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes dense cosine similarity search over indexed chunks with optional metadata filtering.
        """
        if self.vectors is None or len(self.documents) == 0:
            return []

        query_vec = embedding_engine.embed_query(query)
        # Cosine similarity (vectors are L2 normalized)
        similarities = np.dot(self.vectors, query_vec)

        # Apply metadata filters
        candidate_indices = []
        for idx, doc in enumerate(self.documents):
            if filters:
                match = True
                for f_key, f_val in filters.items():
                    if not f_val:
                        continue
                    doc_val = str(doc.get(f_key, "")).lower()
                    if str(f_val).lower() not in doc_val:
                        match = False
                        break
                if not match:
                    continue
            candidate_indices.append(idx)

        if not candidate_indices:
            # Fallback to unfiltered if filters eliminated all chunks
            candidate_indices = list(range(len(self.documents)))

        sub_similarities = [(idx, float(similarities[idx])) for idx in candidate_indices]
        sub_similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in sub_similarities[:top_k]:
            doc_copy = dict(self.documents[idx])
            doc_copy["vector_score"] = round(score, 4)
            results.append(doc_copy)

        return results

    def save_index(self) -> None:
        """Persists document store and embeddings to disk."""
        try:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            with open(self.doc_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=2)
            if self.vectors is not None:
                np.save(str(self.vec_file), self.vectors)
            print(f"[VectorStore] Persisted index to {self.store_dir}")
        except Exception as e:
            print(f"[VectorStore] Error saving index: {e}")

    def load_index(self) -> bool:
        """Loads index from disk if available."""
        try:
            if self.doc_file.exists() and self.vec_file.exists():
                with open(self.doc_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                self.vectors = np.load(str(self.vec_file))
                # Refit embedding engine vocabulary
                texts = [d.get("text", "") for d in self.documents]
                embedding_engine.fit_local_vectorizer(texts)
                print(f"[VectorStore] Loaded {len(self.documents)} documents from existing index.")
                return True
        except Exception as e:
            print(f"[VectorStore] Could not load persisted index: {e}")
        return False

    def count(self) -> int:
        return len(self.documents)


# Global Singleton
vector_store = VectorStore()
