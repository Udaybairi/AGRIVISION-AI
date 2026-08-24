"""
AGRIVISION AI - Visual Disease & Pest Image Indexer and Retriever
Scans the Agriculture Dataset directory tree, catalogs images of crop diseases, pests,
and botanical specimens, and provides semantic and fuzzy image retrieval for the AI Assistant.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.config import AGRICULTURE_DATASET_PATH, PROCESSED_DIR


class ImageRetriever:
    """Indexes and retrieves disease and pest reference images from dataset folders."""

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(self, dataset_root: Optional[Path] = None):
        self.dataset_root = dataset_root or AGRICULTURE_DATASET_PATH
        self.catalog_file = PROCESSED_DIR / "image_catalog.json"
        self.images: List[Dict[str, Any]] = []
        self._load_or_build_catalog()

    def _normalize_name(self, name: str) -> str:
        """Cleans directory and file names into human-readable labels."""
        cleaned = name.replace('_', ' ').replace('-', ' ').replace('(', ' ').replace(')', ' ')
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def build_catalog(self) -> List[Dict[str, Any]]:
        """Scans the dataset directory and creates an index of all images."""
        catalog = []
        if not self.dataset_root or not self.dataset_root.exists():
            print(f"[ImageRetriever] Dataset path not found: {self.dataset_root}")
            return catalog

        try:
            # Walk directory tree
            for root, _, files in os.walk(str(self.dataset_root)):
                root_path = Path(root)
                rel_dir = root_path.relative_to(self.dataset_root)
                dir_parts = [self._normalize_name(p) for p in rel_dir.parts if p]

                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext in self.IMAGE_EXTENSIONS:
                        full_path = root_path / file
                        rel_path = full_path.relative_to(self.dataset_root).as_posix()

                        # Determine category and entities from directory path
                        lower_parts = [p.lower() for p in dir_parts]
                        category = "Crop"
                        if any("pest" in p for p in lower_parts):
                            category = "Pest"
                        elif any("disease" in p or "blight" in p or "rot" in p or "spot" in p or "rust" in p or "mildew" in p for p in lower_parts):
                            category = "Disease"

                        # Extract label
                        label = dir_parts[-1] if dir_parts else Path(file).stem
                        label_clean = self._normalize_name(label)

                        catalog.append({
                            "id": f"img_{len(catalog)}",
                            "file_name": file,
                            "relative_path": rel_path,
                            "full_path": str(full_path),
                            "category": category,
                            "label": label_clean,
                            "path_keywords": " ".join(dir_parts).lower() + " " + Path(file).stem.lower()
                        })

            print(f"[ImageRetriever] Cataloged {len(catalog)} images from {self.dataset_root}")
            self.images = catalog
            self._save_catalog()
        except Exception as e:
            print(f"[ImageRetriever] Error scanning images: {e}")

        return self.images

    def _save_catalog(self) -> None:
        """Saves catalog to disk cache."""
        try:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.catalog_file, "w", encoding="utf-8") as f:
                json.dump(self.images, f, indent=2)
        except Exception as e:
            print(f"[ImageRetriever] Error saving image catalog: {e}")

    def _load_or_build_catalog(self) -> None:
        """Loads cached image catalog or scans directory on first run."""
        if self.catalog_file.exists():
            try:
                with open(self.catalog_file, "r", encoding="utf-8") as f:
                    self.images = json.load(f)
                if self.images:
                    print(f"[ImageRetriever] Loaded {len(self.images)} images from catalog cache.")
                    return
            except Exception as e:
                print(f"[ImageRetriever] Error loading image catalog cache: {e}")

        self.build_catalog()

    def search_images(
        self,
        query: str,
        crop: Optional[str] = None,
        disease: Optional[str] = None,
        pest: Optional[str] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieves visual reference images matching crop, disease, or pest query keywords.
        """
        if not self.images:
            return []

        search_terms = set()
        q_lower = query.lower()
        
        # Add query tokens (filtered)
        STOPWORDS = {
            'what', 'how', 'the', 'and', 'for', 'are', 'is', 'can', 'with', 'from',
            'disease', 'pest', 'plant', 'plants', 'crop', 'crops', 'low', 'money',
            'cost', 'best', 'good', 'grow', 'give', 'help', 'cure', 'need', 'tell'
        }
        for word in re.findall(r'\b[a-zA-Z]{3,}\b', q_lower):
            if word not in STOPWORDS:
                search_terms.add(word)

        if crop:
            search_terms.add(crop.lower())
            for part in crop.lower().split():
                if part not in STOPWORDS:
                    search_terms.add(part)
        if disease:
            search_terms.add(disease.lower())
            for part in disease.lower().split():
                if part not in STOPWORDS:
                    search_terms.add(part)
        if pest:
            search_terms.add(pest.lower())
            for part in pest.lower().split():
                if part not in STOPWORDS:
                    search_terms.add(part)

        if not search_terms:
            return []

        scored_images = []
        for img in self.images:
            keywords = img.get("path_keywords", "")
            score = 0

            # Check matching keywords
            for term in search_terms:
                if term in keywords:
                    score += 2
                    # Exact directory label match bonus
                    if term in img.get("label", "").lower():
                        score += 3

            # Exact disease or pest bonus
            if disease and disease.lower() in keywords:
                score += 5
            if pest and pest.lower() in keywords:
                score += 5
            if crop and crop.lower() in keywords:
                score += 3

            if score > 0:
                scored_images.append((score, img))

        # Sort by score descending
        scored_images.sort(key=lambda x: x[0], reverse=True)

        results = []
        seen_labels = set()
        for score, img in scored_images:
            # Pick a diverse set of representative images
            rel_path = img.get("relative_path", "")
            url = f"/api/dataset-image/{rel_path}"
            
            item = {
                "id": img.get("id"),
                "label": img.get("label"),
                "category": img.get("category"),
                "image_url": url,
                "relative_path": rel_path,
                "score": score
            }
            results.append(item)
            if len(results) >= top_k:
                break

        return results


# Global Singleton
image_retriever = ImageRetriever()
