"""
config.py — Configuration centralisée du projet
Real-Time Sentiment Intelligence Platform
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# ── Helper pour récupérer les secrets (.env, os.environ ou st.secrets) ──
def get_secret(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default

# ── Chemins ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATABASE_PATH = PROJECT_ROOT / "data" / "sentiment.db"

# ── Reddit API ───────────────────────────────────────────
REDDIT_CLIENT_ID = get_secret("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = get_secret("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = get_secret("REDDIT_USER_AGENT", "sentiment-intelligence/1.0")

# ── NewsAPI ──────────────────────────────────────────────
NEWSAPI_KEY = get_secret("NEWSAPI_KEY", "")

# ── Groq API (LLM cloud gratuit) ────────────────────────
GROQ_API_KEY = get_secret("GROQ_API_KEY", "")

# ── Modèles NLP ─────────────────────────────────────────
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistral"  # Pour Ollama

# ── Redis ────────────────────────────────────────────────
REDIS_URL = get_secret("REDIS_URL", "")
REDIS_HOST = get_secret("REDIS_HOST", "localhost")
REDIS_PORT = int(get_secret("REDIS_PORT", "6379") or 6379)
REDIS_PASSWORD = get_secret("REDIS_PASSWORD", "")
REDIS_SSL = get_secret("REDIS_SSL", "False").lower() in ("true", "1", "t")
REDIS_STREAM_NAME = "sentiment_stream"
