import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import io
from PIL import Image
import numpy as np
import pytest

from backend.ml.crop_recommendation import crop_recommender
from backend.ml.fertilizer_recommendation import fertilizer_advisor
from backend.ml.disease_prediction import disease_predictor
from backend.ml.pest_detection import pest_detector


def test_crop_recommendation_ml():
    """Verifies that ML crop recommender outputs valid crop candidates and confidence."""
    # Test conditions suitable for rice (high rainfall, high humidity)
    res = crop_recommender.predict(
        nitrogen=80, phosphorus=40, potassium=40,
        temperature=28.0, humidity=82.0, ph=6.5, rainfall=220.0
    )
    assert res is not None
    assert "primary_crop" in res
    assert res["confidence"] > 0.0
    assert len(res["recommendations"]) >= 1
    assert "profile" in res


def test_fertilizer_advisor():
    """Verifies nutrient deficiency diagnosis and corrective actions."""
    # Test low nitrogen scenario for Rice
    res = fertilizer_advisor.recommend(
        crop_name="Rice", nitrogen=20, phosphorus=50, potassium=40, soil_type="Clayey"
    )
    assert res is not None
    assert res["crop"] == "Rice"
    assert "Deficiency" in res["status"] or "Excess" in res["status"] or "Optimal" in res["status"]
    assert "recommended_action" in res
    assert "nutrient_comparison" in res


def test_disease_predictor_with_synthetic_leaf():
    """Verifies PyTorch ResNet9 inference on synthetic RGB image."""
    img = Image.new('RGB', (256, 256), color=(40, 180, 50))
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    image_bytes = byte_arr.getvalue()

    result = disease_predictor.predict(image_bytes, top_k=3)
    assert result is not None
    assert "crop" in result
    assert "disease" in result
    assert "confidence" in result
    assert len(result["top_candidates"]) <= 3
    assert "diagnostics" in result


def test_pest_detector():
    """Verifies pest IPM advice for common crop pests."""
    res = pest_detector.diagnose_from_text("whitefly on cotton leaves with yellow mosaic")
    assert res is not None
    assert "Whitefly" in res["identified_pest"]
    assert "ipm_management" in res
    assert "cultural_control" in res["ipm_management"]
    assert "biological_control" in res["ipm_management"]
