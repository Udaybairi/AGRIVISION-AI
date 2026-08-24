"""
AGRIVISION AI - Machine Learning & Computer Vision Subsystem
"""

from backend.ml.crop_recommendation import crop_recommender, CropRecommender
from backend.ml.fertilizer_recommendation import fertilizer_advisor, FertilizerAdvisor
from backend.ml.disease_prediction import disease_predictor, DiseasePredictor
from backend.ml.pest_detection import pest_detector, PestDetector

__all__ = [
    "crop_recommender",
    "CropRecommender",
    "fertilizer_advisor",
    "FertilizerAdvisor",
    "disease_predictor",
    "DiseasePredictor",
    "pest_detector",
    "PestDetector"
]
