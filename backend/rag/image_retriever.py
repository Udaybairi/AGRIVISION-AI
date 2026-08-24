"""
AGRIVISION AI - Visual Disease & Pest Image Indexer and Retriever
Scans the Agriculture Dataset directory tree, catalogs images of crop diseases, pests,
and botanical specimens, and provides semantic, fuzzy, and high-resolution photographic image retrieval.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.config import AGRICULTURE_DATASET_PATH, PROCESSED_DIR


# Curated high-resolution agricultural photographic references (for cloud & production)
CURATED_AGRI_PHOTOS: Dict[str, List[Dict[str, str]]] = {
    "cotton": [
        {
            "label": "Cotton Boll Specimen",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1594488554271-e0e648c66e2c?auto=format&fit=crop&w=800&q=80"
        },
        {
            "label": "Cotton Field Plantation",
            "category": "Cultivation Field",
            "url": "https://images.unsplash.com/photo-1605000797499-95a51c5269ae?auto=format&fit=crop&w=800&q=80"
        },
        {
            "label": "Cotton Leaf Foliage",
            "category": "Botanical Diagnostic",
            "url": "https://images.unsplash.com/photo-1586771107445-d3ca888129ff?auto=format&fit=crop&w=800&q=80"
        },
        {
            "label": "Bollworm & Pest Sentinel",
            "category": "Pest Management",
            "url": "https://images.unsplash.com/photo-1585250004683-4f65a88a745e?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "rice": [
        {
            "label": "Rice Paddy Field",
            "category": "Cultivation Field",
            "url": "https://images.unsplash.com/photo-1536657464919-892534f60d6e?auto=format&fit=crop&w=800&q=80"
        },
        {
            "label": "Golden Grain Ears",
            "category": "Harvest Reference",
            "url": "https://images.unsplash.com/photo-1586201375761-83865001e31c?auto=format&fit=crop&w=800&q=80"
        },
        {
            "label": "Rice Stem & Leaf Health",
            "category": "Agronomic Diagnostic",
            "url": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "wheat": [
        {
            "label": "Golden Wheat Canopy",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?auto=format&fit=crop&w=800&q=80"
        },
        {
            "label": "Wheat Field Ecology",
            "category": "Cultivation Field",
            "url": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "tomato": [
        {
            "label": "Vine Tomatoes & Leaf Canopy",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1592924357228-91a4daadcfea?auto=format&fit=crop&w=800&q=80"
        },
        {
            "label": "Tomato Plant Diagnostic",
            "category": "Botanical Diagnostic",
            "url": "https://images.unsplash.com/photo-1546470427-e26264be0b11?auto=format&fit=crop&w=800&q=80"
        },
        {
            "label": "Early Blight & Foliar Symptoms",
            "category": "Disease Diagnostic",
            "url": "https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "maize": [
        {
            "label": "Corn Ear on Stalk",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1551754655-cd27e38d2076?auto=format&fit=crop&w=800&q=80"
        },
        {
            "label": "Maize Canopy Field",
            "category": "Cultivation Field",
            "url": "https://images.unsplash.com/photo-1597916829826-02e5bb4a54e0?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "corn": [
        {
            "label": "Corn Ear on Stalk",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1551754655-cd27e38d2076?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "potato": [
        {
            "label": "Potato Foliage & Harvest",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1518977676601-b53f82aba655?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "banana": [
        {
            "label": "Banana Plantation Tree",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "mango": [
        {
            "label": "Mango Orchard Canopy",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1553279768-865429fa0078?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "grapes": [
        {
            "label": "Vineyard Grapes",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1537640538966-79f369143f8f?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "apple": [
        {
            "label": "Apple Orchard Trees",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "orange": [
        {
            "label": "Citrus Grove",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1582979512210-99b6a53386f9?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "coffee": [
        {
            "label": "Coffee Cherries & Foliage",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "papaya": [
        {
            "label": "Papaya Tree & Fruit",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1517282009859-f000ec3b26fe?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "watermelon": [
        {
            "label": "Watermelon Crop",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1587049352846-4a222e784d38?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "pomegranate": [
        {
            "label": "Pomegranate Tree",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1615485290382-441e4d049cb5?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "coconut": [
        {
            "label": "Coconut Palm Grove",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1544551763-46a013bb70d5?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "pepper": [
        {
            "label": "Bell Pepper Plants",
            "category": "Crop Specimen",
            "url": "https://images.unsplash.com/photo-1563565375-f3fdfdbefa83?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "aphid": [
        {
            "label": "Aphid Pest Colony",
            "category": "Pest Specimen",
            "url": "https://images.unsplash.com/photo-1585250004683-4f65a88a745e?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "pest": [
        {
            "label": "Insect Pest Macro",
            "category": "Pest Specimen",
            "url": "https://images.unsplash.com/photo-1585250004683-4f65a88a745e?auto=format&fit=crop&w=800&q=80"
        }
    ],
    "caterpillar": [
        {
            "label": "Foliar Caterpillar",
            "category": "Pest Specimen",
            "url": "https://images.unsplash.com/photo-1534067783941-51c9c23ecefd?auto=format&fit=crop&w=800&q=80"
        }
    ]
}


class ImageRetriever:
    """Indexes and retrieves disease and pest reference images from dataset folders or curated CDNs."""

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
            return catalog

        try:
            for root, _, files in os.walk(str(self.dataset_root)):
                root_path = Path(root)
                rel_dir = root_path.relative_to(self.dataset_root)
                dir_parts = [self._normalize_name(p) for p in rel_dir.parts if p]

                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext in self.IMAGE_EXTENSIONS:
                        full_path = root_path / file
                        rel_path = full_path.relative_to(self.dataset_root).as_posix()

                        lower_parts = [p.lower() for p in dir_parts]
                        category = "Crop"
                        if any("pest" in p for p in lower_parts):
                            category = "Pest"
                        elif any("disease" in p or "blight" in p or "rot" in p or "spot" in p or "rust" in p or "mildew" in p for p in lower_parts):
                            category = "Disease"

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
                    return
            except Exception as e:
                print(f"[ImageRetriever] Could not load image catalog: {e}")

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
        Guarantees diverse, high-resolution photographic images without duplicates.
        """
        target_entity = (crop or pest or disease or "").lower().strip()
        q_lower = query.lower()

        # 1. Check curated high-resolution photo dictionary first for photographic quality
        for key, photos in CURATED_AGRI_PHOTOS.items():
            if key in target_entity or key in q_lower or (crop and key in crop.lower()) or (pest and key in pest.lower()):
                results = []
                for idx, photo in enumerate(photos[:top_k]):
                    results.append({
                        "id": f"curated_{key}_{idx}",
                        "label": photo["label"],
                        "category": photo["category"],
                        "image_url": photo["url"],
                        "relative_path": f"curated/{key}/{idx}",
                        "score": 10
                    })
                return results

        if not self.images:
            return []

        search_terms = set()
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
        if disease:
            search_terms.add(disease.lower())
        if pest:
            search_terms.add(pest.lower())

        if not search_terms:
            return []

        scored_images = []
        for img in self.images:
            keywords = img.get("path_keywords", "")
            score = 0

            for term in search_terms:
                if term in keywords:
                    score += 2
                    if term in img.get("label", "").lower():
                        score += 3

            if disease and disease.lower() in keywords:
                score += 5
            if pest and pest.lower() in keywords:
                score += 5
            if crop and crop.lower() in keywords:
                score += 3

            if score > 0:
                scored_images.append((score, img))

        scored_images.sort(key=lambda x: x[0], reverse=True)

        results = []
        seen_dirs = set()
        for score, img in scored_images:
            rel_path = img.get("relative_path", "")
            # Deduplicate by parent directory so we get diverse visual specimens
            parent_dir = str(Path(rel_path).parent)
            if parent_dir in seen_dirs and len(seen_dirs) < len(scored_images):
                continue
            seen_dirs.add(parent_dir)

            url = f"/api/dataset-image/{rel_path}"
            results.append({
                "id": img.get("id"),
                "label": img.get("label"),
                "category": img.get("category"),
                "image_url": url,
                "relative_path": rel_path,
                "score": score
            })
            if len(results) >= top_k:
                break

        return results


# Global Singleton
image_retriever = ImageRetriever()
