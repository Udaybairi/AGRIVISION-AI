"""
AGRIVISION AI - Intelligent Query Router
Routes user queries to specialized retrieval pipelines, ML models, or knowledge domains.
"""

import re
from typing import Dict, Any


ROUTER_CATEGORIES = [
    "Crop Recommendation",
    "Fertilizer Recommendation",
    "Disease Identification",
    "Disease Management",
    "Pest Identification",
    "Pest Management",
    "Soil Health",
    "Crop Cultivation",
    "Irrigation",
    "Weather-related agriculture",
    "Dataset Knowledge",
    "General Agriculture"
]


class QueryRouter:
    """Categorizes queries and assigns processing pipelines and metadata filtering strategies."""

    def route(self, query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        lower = query.lower()

        # Intent Classification Rules
        if any(w in lower for w in ['recommend crop', 'which crop', 'best crop', 'what to grow', 'suitable crop']):
            category = "Crop Recommendation"
            pipeline = "ml_crop_rag_hybrid"
        elif any(w in lower for w in ['fertilizer', 'urea', 'dap', 'mop', 'npk', 'nitrogen deficiency', 'phosphorus deficiency', 'potassium deficiency']):
            category = "Fertilizer Recommendation"
            pipeline = "fertilizer_rag_advisor"
        elif any(w in lower for w in ['identify disease', 'what disease', 'leaf spot', 'blight', 'scab', 'rust', 'rot', 'yellow leaf curl']):
            category = "Disease Identification" if 'what' in lower or 'identify' in lower else "Disease Management"
            pipeline = "disease_knowledge_retrieval"
        elif any(w in lower for w in ['pest', 'insect', 'caterpillar', 'aphid', 'whitefly', 'bollworm', 'stem borer', 'mite', 'thrips']):
            category = "Pest Identification" if 'identify' in lower or 'what insect' in lower else "Pest Management"
            pipeline = "pest_ipm_retrieval"
        elif any(w in lower for w in ['soil ph', 'soil type', 'loam', 'clay soil', 'salinity', 'organic carbon']):
            category = "Soil Health"
            pipeline = "soil_health_retrieval"
        elif any(w in lower for w in ['irrigation', 'watering', 'drip', 'sprinkler', 'water requirement']):
            category = "Irrigation"
            pipeline = "irrigation_retrieval"
        elif any(w in lower for w in ['weather', 'rainfall', 'monsoon', 'temperature', 'frost', 'drought']):
            category = "Weather-related agriculture"
            pipeline = "weather_agriculture_retrieval"
        elif any(w in lower for w in ['dataset', 'image archive', 'pdf manual', 'excel knowledge base']):
            category = "Dataset Knowledge"
            pipeline = "dataset_direct_retrieval"
        else:
            category = "General Agriculture"
            pipeline = "general_rag_hybrid"

        return {
            'category': category,
            'pipeline': pipeline,
            'confidence': 0.94,
            'target_crop': metadata.get('crop'),
            'target_entity': metadata.get('disease') or metadata.get('pest') or metadata.get('nutrient')
        }


# Global Singleton
query_router = QueryRouter()
