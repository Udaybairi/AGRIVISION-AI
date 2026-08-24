"""
AGRIVISION AI - Application & REST API Server
"Intelligent Farming. Evidence-Based Decisions."
"""

import os
import sys
import io
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Ensure project root is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import (
    HOST, PORT, DEBUG, SECRET_KEY,
    PROJECT_NAME, PROJECT_TAGLINE,
    AGRICULTURE_DATASET_PATH
)
from backend.ml.crop_recommendation import crop_recommender
from backend.ml.fertilizer_recommendation import fertilizer_advisor
from backend.ml.disease_prediction import disease_predictor
from backend.ml.pest_detection import pest_detector
from backend.rag.pipeline import advanced_rag_pipeline
from backend.rag.vector_store import vector_store

# Initialize Flask App with custom template and static folders
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "frontend" / "templates"),
    static_folder=str(BASE_DIR / "frontend" / "static")
)
app.config["SECRET_KEY"] = SECRET_KEY
CORS(app)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


# ============================================================================
# PAGE ROUTES (3D Frontend)
# ============================================================================

@app.route('/')
def home():
    """Renders the AGRIVISION AI Futuristic 3D Agriculture Platform."""
    return render_template(
        'index.html',
        project_name=PROJECT_NAME,
        tagline=PROJECT_TAGLINE,
        dataset_count=vector_store.count()
    )


# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
@app.route('/api/stats', methods=['GET'])
def health_check():
    """System health check and status reporting."""
    return jsonify({
        "status": "healthy",
        "platform": PROJECT_NAME,
        "tagline": PROJECT_TAGLINE,
        "dataset_path": str(AGRICULTURE_DATASET_PATH),
        "indexed_documents": vector_store.count(),
        "ml_models": {
            "crop_recommendation": crop_recommender.model is not None,
            "disease_prediction": disease_predictor.model is not None,
            "fertilizer_advisor": fertilizer_advisor.df is not None,
            "pest_detector": True
        }
    })


@app.route('/api/chat', methods=['POST'])
def rag_chat():
    """
    Advanced RAG Conversation Endpoint.
    Executes: Query Rewriting -> Multi-Query -> Routing -> Hybrid Search -> Reranking -> LLM -> Citations.
    """
    try:
        data = request.get_json(silent=True) or (request.form.to_dict() if request.form else {}) or {}
        user_query = data.get("query", "").strip()
        custom_filters = data.get("filters", {})

        if not user_query:
            return jsonify({"error": "Empty query received"}), 400

        result = advanced_rag_pipeline.process_query(user_query, custom_filters=custom_filters)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Chat API error: {e}", exc_info=True)
        return jsonify({"error": f"Internal RAG error: {str(e)}"}), 500


@app.route('/api/crop-recommend', methods=['POST'])
def crop_recommendation_api():
    """
    ML Crop Recommendation + Grounded RAG Evidence Synthesis.
    """
    try:
        data = request.get_json(silent=True) or (request.form.to_dict() if request.form else {}) or {}
        
        # If form data was sent
        if not data and request.form:
            data = {
                "nitrogen": float(request.form.get("nitrogen", 50)),
                "phosphorus": float(request.form.get("phosphorus", 50)),
                "potassium": float(request.form.get("potassium", 50)),
                "temperature": float(request.form.get("temperature", 25.0)),
                "humidity": float(request.form.get("humidity", 70.0)),
                "ph": float(request.form.get("ph", 6.5)),
                "rainfall": float(request.form.get("rainfall", 100.0)),
                "city": request.form.get("city", "")
            }

        nitrogen = float(data.get("nitrogen", 50))
        phosphorus = float(data.get("phosphorus", 50))
        potassium = float(data.get("potassium", 50))
        temperature = float(data.get("temperature", 25.0))
        humidity = float(data.get("humidity", 70.0))
        ph = float(data.get("ph", 6.5))
        rainfall = float(data.get("rainfall", 100.0))
        city = data.get("city", "").strip()

        # Optional Weather fetch if city provided
        if city:
            weather = crop_recommender.fetch_weather_data(city)
            if weather:
                temperature = weather["temperature"]
                humidity = weather["humidity"]

        # Run ML inference
        prediction_result = crop_recommender.predict(
            nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall
        )

        primary_crop = prediction_result["primary_crop"]

        # Retrieve RAG evidence regarding best agronomic practices for recommended crop
        rag_query = f"Cultivation guidelines, soil requirements, and nutrient management for {primary_crop}"
        rag_evidence = advanced_rag_pipeline.process_query(
            rag_query, custom_filters={"crop": primary_crop}
        )

        return jsonify({
            "ml_prediction": prediction_result,
            "agronomic_evidence": {
                "answer": rag_evidence["answer"],
                "citations": rag_evidence["evidence_items"],
                "grounding_level": rag_evidence["grounding_level"],
                "grounding_score": rag_evidence["grounding_score"]
            }
        })
    except Exception as e:
        app.logger.error(f"Crop recommendation API error: {e}", exc_info=True)
        return jsonify({"error": f"Crop prediction error: {str(e)}"}), 500


@app.route('/api/fertilizer-recommend', methods=['POST'])
def fertilizer_recommendation_api():
    """
    Fertilizer & Soil Health Advisor with RAG Knowledge Verification.
    """
    try:
        data = request.get_json(silent=True) or (request.form.to_dict() if request.form else {}) or {}
        if not data and request.form:
            data = {
                "crop": request.form.get("crop", "Rice"),
                "nitrogen": float(request.form.get("nitrogen", 50)),
                "phosphorus": float(request.form.get("phosphorus", 50)),
                "potassium": float(request.form.get("potassium", 50)),
                "soil_type": request.form.get("soil_type", "Loamy"),
                "ph": float(request.form.get("ph", 6.5)) if request.form.get("ph") else None
            }

        crop = str(data.get("crop", "Rice"))
        n = float(data.get("nitrogen", 50))
        p = float(data.get("phosphorus", 50))
        k = float(data.get("potassium", 50))
        soil_type = str(data.get("soil_type", "Loamy"))
        ph = float(data.get("ph")) if data.get("ph") is not None else None

        advisor_result = fertilizer_advisor.recommend(
            crop_name=crop, nitrogen=n, phosphorus=p, potassium=k, soil_type=soil_type, ph=ph
        )

        # Retrieve RAG evidence on fertilizer management
        rag_query = f"Recommended fertilizer schedule and soil nutrient management for {crop} {advisor_result['status']}"
        rag_evidence = advanced_rag_pipeline.process_query(
            rag_query, custom_filters={"crop": crop}
        )

        return jsonify({
            "advisory": advisor_result,
            "agronomic_evidence": {
                "answer": rag_evidence["answer"],
                "citations": rag_evidence["evidence_items"],
                "grounding_level": rag_evidence["grounding_level"]
            }
        })
    except Exception as e:
        app.logger.error(f"Fertilizer API error: {e}", exc_info=True)
        return jsonify({"error": f"Fertilizer advisor error: {str(e)}"}), 500


@app.route('/api/disease-predict', methods=['POST'])
def disease_prediction_api():
    """
    Plant Doctor: Computer Vision Leaf Diagnosis + RAG Grounded Management.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No image file provided in 'file' field"}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format. Upload PNG, JPG, JPEG, or WEBP"}), 400

        image_bytes = file.read()
        cv_result = disease_predictor.predict(image_bytes, top_k=3)

        # Grounded RAG Retrieval for the predicted disease
        crop = cv_result["crop"]
        disease = cv_result["disease"]
        is_healthy = cv_result["is_healthy"]

        if is_healthy:
            rag_query = f"Best maintenance practices and growth requirements for healthy {crop}"
        else:
            rag_query = f"Symptoms, etiology, and evidence-based integrated pest and disease management for {crop} {disease}"

        rag_evidence = advanced_rag_pipeline.process_query(
            rag_query, custom_filters={"crop": crop, "disease": disease if not is_healthy else None}
        )

        return jsonify({
            "cv_diagnosis": cv_result,
            "rag_evidence": {
                "answer": rag_evidence["answer"],
                "citations": rag_evidence["evidence_items"],
                "grounding_level": rag_evidence["grounding_level"],
                "grounding_score": rag_evidence["grounding_score"],
                "trace": rag_evidence.get("trace")
            }
        })
    except Exception as e:
        app.logger.error(f"Disease prediction API error: {e}", exc_info=True)
        return jsonify({"error": f"Disease diagnosis error: {str(e)}"}), 500


@app.route('/api/pest-detect', methods=['POST'])
def pest_detection_api():
    """
    Pest AI: Identification & Integrated Pest Management (IPM).
    """
    try:
        data = request.get_json(silent=True) or (request.form.to_dict() if request.form else {}) or {}
        description = (
            data.get("description", "") or
            data.get("pest_name", "") or
            data.get("pest", "") or
            data.get("query", "")
        ).strip()

        if not description:
            return jsonify({"error": "Please provide a pest description or name"}), 400

        pest_result = pest_detector.diagnose_from_text(description)
        pest_name = pest_result["identified_pest"]

        rag_query = f"Integrated pest management and biological control for {pest_name}"
        rag_evidence = advanced_rag_pipeline.process_query(rag_query)

        return jsonify({
            "pest_diagnosis": pest_result,
            "rag_evidence": {
                "answer": rag_evidence["answer"],
                "citations": rag_evidence["evidence_items"],
                "grounding_level": rag_evidence["grounding_level"]
            }
        })
    except Exception as e:
        app.logger.error(f"Pest API error: {e}", exc_info=True)
        return jsonify({"error": f"Pest diagnostic error: {str(e)}"}), 500


@app.route('/api/observability', methods=['GET'])
def observability_api():
    """
    RAG Observability & Trace Visualizer Endpoint for Evaluators.
    """
    sample_query = request.args.get("query", "tomato leaf black what medicine")
    result = advanced_rag_pipeline.process_query(sample_query)
    return jsonify({
        "sample_query": sample_query,
        "trace": result["trace"],
        "query_transformation": result["query_transformation"],
        "category": result["category"],
        "evidence_count": len(result["evidence_items"]),
        "citations": result["evidence_items"],
        "grounding_level": result["grounding_level"],
        "grounding_score": result["grounding_score"],
        "latency_ms": result["latency_ms"]
    })


@app.route('/api/dataset-image/<path:image_path>', methods=['GET'])
def get_dataset_image(image_path: str):
    """
    Streams dataset images safely from the Agriculture Dataset directory.
    """
    try:
        dataset_dir = Path(AGRICULTURE_DATASET_PATH).resolve()
        target_file = (dataset_dir / image_path).resolve()
        
        # Security check: ensure target file is within dataset_dir
        if not str(target_file).startswith(str(dataset_dir)) or not target_file.exists():
            return jsonify({"error": "Image not found"}), 404
            
        return send_from_directory(target_file.parent, target_file.name)
    except Exception as e:
        app.logger.error(f"Image delivery error: {e}")
        return jsonify({"error": "Could not load image"}), 500


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print(f" {PROJECT_NAME} — {PROJECT_TAGLINE}")
    print("=" * 70)
    print(f" * Server running on: http://{HOST}:{PORT}")
    print(f" * Dataset path: {AGRICULTURE_DATASET_PATH}")
    print(f" * Total indexed knowledge chunks: {vector_store.count()}")
    print("=" * 70)
    app.run(host=HOST, port=PORT, debug=DEBUG)
