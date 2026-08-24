"""
AGRIVISION AI - Centralized Configuration System
"Intelligent Farming. Evidence-Based Decisions."
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Project Metadata
PROJECT_NAME = "AGRIVISION AI"
PROJECT_TAGLINE = "Intelligent Farming. Evidence-Based Decisions."
PROJECT_VERSION = "2.0.0"

# Dataset Configuration (Do not hard-code paths across the app)
DEFAULT_DATASET_PATH = r"C:\Users\uday kumar\Downloads\Agriculture Dataset"
AGRICULTURE_DATASET_PATH = Path(
    os.getenv("AGRICULTURE_DATASET_PATH", DEFAULT_DATASET_PATH)
).resolve()

# Workspace Storage Paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Ensure runtime directories exist
for directory in [DATA_DIR, MODELS_DIR, PROCESSED_DIR, VECTOR_STORE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Pre-trained ML Models
CROP_MODEL_PATH = MODELS_DIR / "RandomForest.pkl"
DISEASE_MODEL_PATH = MODELS_DIR / "plant_disease_model.pth"
FERTILIZER_CSV_PATH = PROCESSED_DIR / "fertilizer.csv"

# API Keys & Services
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")


# LLM & RAG Configuration
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto") # "gemini", "groq", "openai", "auto", "local"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto") # "openai", "gemini", "local"
RAG_HYBRID_ALPHA = float(os.getenv("RAG_HYBRID_ALPHA", "0.6")) # Weight for vector search vs BM25
RETRIEVAL_CANDIDATE_LIMIT = int(os.getenv("RETRIEVAL_CANDIDATE_LIMIT", "25"))
RERANKED_EVIDENCE_LIMIT = int(os.getenv("RERANKED_EVIDENCE_LIMIT", "6"))
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.35"))

# Server Configuration
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
SECRET_KEY = os.getenv("SECRET_KEY", "agrivision_ai_secure_token_2026_super_secret")
