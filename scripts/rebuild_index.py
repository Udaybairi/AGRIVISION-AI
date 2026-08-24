"""
AGRIVISION AI - Rebuild Vector Index Utility
Forces re-indexing of all agriculture documents and updates vector embeddings.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import AGRICULTURE_DATASET_PATH, VECTOR_STORE_DIR
from backend.rag.ingestion import data_ingestion
from backend.rag.vector_store import vector_store


def main():
    print(f"Rebuilding index from {AGRICULTURE_DATASET_PATH}...")
    chunks = data_ingestion.run_full_ingestion()
    vector_store.add_documents(chunks)
    print(f"Index successfully rebuilt. Total entries: {vector_store.count()}")


if __name__ == '__main__':
    main()
