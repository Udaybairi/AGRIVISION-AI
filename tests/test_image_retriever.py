"""
Unit tests for AGRIVISION AI Visual Disease and Pest Image Retriever.
"""

import pytest
from backend.rag.image_retriever import image_retriever


def test_image_catalog_loading():
    """Verify that images are cataloged from the dataset."""
    assert len(image_retriever.images) > 0, "Image catalog should contain indexed images"


def test_crop_image_search():
    """Verify image retrieval for crops like tomato."""
    results = image_retriever.search_images("tomato", crop="Tomato", top_k=3)
    assert len(results) > 0
    assert any("tomato" in r["label"].lower() or "tomato" in r["relative_path"].lower() for r in results)
    assert results[0]["image_url"].startswith("/api/dataset-image/")


def test_pest_image_search():
    """Verify image retrieval for pests like aphids."""
    results = image_retriever.search_images("aphid damage", pest="Aphid", top_k=3)
    assert len(results) > 0
    assert any("aphid" in r["label"].lower() or "aphid" in r["relative_path"].lower() for r in results)


def test_empty_query():
    """Verify graceful handling of unrelated or empty queries."""
    results = image_retriever.search_images("", top_k=2)
    assert results == []
