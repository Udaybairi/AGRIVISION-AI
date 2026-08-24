"""
AGRIVISION AI - Dataset Ingestion & Vector Index Builder Script
Scans the configured Agriculture Dataset root, extracts multi-format content, and persists vector embeddings.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import AGRICULTURE_DATASET_PATH, VECTOR_STORE_DIR
from backend.rag.ingestion import data_ingestion
from backend.rag.vector_store import vector_store


def main():
    print("=" * 70)
    print(" AGRIVISION AI — Knowledge Ingestion & Vector Indexing")
    print("=" * 70)
    print(f"Dataset Path: {AGRICULTURE_DATASET_PATH}")
    print(f"Vector Store Directory: {VECTOR_STORE_DIR}")

    # Run full ingestion
    chunks = data_ingestion.run_full_ingestion()
    print(f"\n[+] Total chunks extracted: {len(chunks)}")

    # Index into vector store
    print("\n[+] Computing dense semantic embeddings and building vector index...")
    vector_store.add_documents(chunks)

    print(f"\n[OK] Indexing completed successfully. Total indexed documents: {vector_store.count()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
