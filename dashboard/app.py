"""
app.py — Real-Time Sentiment Intelligence Platform
Dashboard interactif Streamlit avec streaming Redis, analyse RoBERTa et Groq LLM
"""

import sys
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configurer stdout en UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.storage.database import SessionLocal, Post, CrisisAlert, init_db
from src.streaming.redis_stream import broker
from src.ingestion.reddit_collector import reddit_collector
from src.ingestion.news_collector import news_collector
import importlib
from src.ingestion.simulator import generate_single_event
from src.ingestion.topic_search import search_and_analyze_topic
import src.nlp.llm_summary
importlib.reload(src.nlp.llm_summary)
from src.nlp.llm_summary import generate_crisis_report, generate_topic_deep_dive

# ── Configuration de la page Streamlit ───────────────────
st.set_page_config(
    page_title="Sentiment Analysis — Plateforme NLP/LLM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS moderne & typographie calligraphique
st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Sidebar Navigation spécifique */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        display: none !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] {
        margin-top: -6px;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioGroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"] {
        display: flex !important;
        align-items: center !important;
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 10px 16px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    /* Cacher complètement le cercle et point radio dans la navigation sidebar */
    [data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"] div:first-child:not([data-testid="stMarkdownContainer"]),
    [data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"] .e1mpz0hj4,
    [data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"] .e1mpz0hj5,
    [data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"] input[type="radio"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"]:hover {
        background: #273549 !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        transform: translateX(2px) !important;
    }
    /* Option sélectionnée dans la navigation */
    [data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"],
    [data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"]:has(input:checked) {
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.35), rgba(99, 102, 241, 0.4)) !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25) !important;
    }
    /* Vraies icônes vectorielles SVG pour chaque bouton de navigation */
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > div:nth-child(1) [data-testid="stRadioOption"]::before {
        content: "";
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 10px;
        flex-shrink: 0;
        background-color: currentColor;
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E") no-repeat center;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E") no-repeat center;
    }
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > div:nth-child(2) [data-testid="stRadioOption"]::before {
        content: "";
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 10px;
        flex-shrink: 0;
        background-color: currentColor;
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3C/svg%3E") no-repeat center;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3C/svg%3E") no-repeat center;
    }
    [data-testid="stSidebar"] div[data-testid="stRadioGroup"] > div:nth-child(3) [data-testid="stRadioOption"]::before {
        content: "";
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 10px;
        flex-shrink: 0;
        background-color: currentColor;
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E") no-repeat center;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E") no-repeat center;
    }

    /* Radios et contrôles horizontaux dans la zone principale */
    div[data-testid="stRadio"] [data-testid="stRadioGroup"][data-orientation="horizontal"],
    div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        gap: 12px !important;
    }
    div[data-testid="stRadio"] [data-testid="stRadioGroup"][data-orientation="horizontal"] > div,
    div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > div {
        display: inline-flex !important;
        width: auto !important;
    }
    div[data-testid="stRadio"] [data-testid="stRadioGroup"][data-orientation="horizontal"] [data-testid="stRadioOption"],
    div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] label {
        display: inline-flex !important;
        align-items: center !important;
        width: auto !important;
        padding: 6px 14px !important;
        border-radius: 8px !important;
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #cbd5e1 !important;
        cursor: pointer !important;
    }

    /* Segmented Control / Sélecteurs horizontaux */
    div[data-testid="stSegmentedControl"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 6px !important;
    }
    div[data-testid="stSegmentedControl"] button {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        border-radius: 8px !important;
        padding: 6px 14px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stSegmentedControl"] button:hover {
        background: #273549 !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[data-checked="true"] {
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.4), rgba(99, 102, 241, 0.45)) !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
        box-shadow: 0 2px 10px rgba(56, 189, 248, 0.25) !important;
    }


    /* Carte Statut Infrastructure */
    .infra-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .infra-header {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #38bdf8;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e293b;
    }
    .infra-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid rgba(51, 65, 85, 0.3);
    }
    .infra-row:last-child {
        border-bottom: none;
    }
    .infra-label {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #94a3b8;
        font-size: 12px;
        font-weight: 500;
    }
    .infra-badge-online {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .infra-badge-online::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #22c55e;
        display: inline-block;
        box-shadow: 0 0 6px #22c55e;
    }
    .infra-badge-offline {
        background: rgba(234, 179, 8, 0.15);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.3);
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    .infra-badge-val {
        background: #1e293b;
        color: #f1f5f9;
        border: 1px solid #334155;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        font-family: monospace;
    }
    .infra-badge-model {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    .infra-badge-llm {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }

    /* Badges de Sentiment Typographiques (Sans émojis) */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-pos {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .badge-neg {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .badge-neu {
        background: rgba(59, 130, 246, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }

    .kpi-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 4px;
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
    }
    .action-box {
        background: #1e293b;
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .post-card {
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        background: #1e293b;
        border-left: 5px solid #64748b;
    }
    .post-card.pos { border-left-color: #22c55e; }
    .post-card.neg { border-left-color: #ef4444; }
    .post-card.neu { border-left-color: #3b82f6; }
    
    /* Hero Accueil Chaleureux & Pro */
    .presentation-hero {
        text-align: center;
        padding: 20px 20px 10px 20px;
        position: relative;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 5px 16px;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(56, 189, 248, 0.15));
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 9999px;
        color: #fbbf24;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }
    .presentation-title {
        font-size: 44px;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 30%, #38bdf8 70%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .presentation-subtitle {
        font-size: 18px;
        color: #94a3b8;
        font-weight: 400;
        max-width: 820px;
        margin: 0 auto 20px auto;
        line-height: 1.6;
    }
    .hero-stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        max-width: 920px;
        margin: 0 auto 24px auto;
    }
    .hero-stat-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 12px;
        padding: 12px 10px;
        text-align: center;
        transition: all 0.25s ease;
    }
    .hero-stat-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(56, 189, 248, 0.15);
    }
    .hero-stat-val {
        font-size: 20px;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-stat-lbl {
        font-size: 11px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 2px;
    }

    /* Piliers Chaleureux & Pro */
    .pillar-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 22px;
        height: 100%;
        transition: all 0.3s ease;
    }
    .pillar-card:hover {
        border-color: #38bdf8;
        transform: translateY(-3px);
        box-shadow: 0 12px 24px -6px rgba(56, 189, 248, 0.15);
    }
    .pillar-icon-box {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
    }
    .pillar-title {
        font-size: 17px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 8px;
    }
    .pillar-desc {
        font-size: 14px;
        color: #94a3b8;
        line-height: 1.6;
    }

    /* Pipeline Pas-à-Pas */
    .pipeline-step-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        height: 100%;
        transition: all 0.25s ease;
    }
    .pipeline-step-card:hover {
        border-color: #818cf8;
        transform: translateY(-2px);
    }

    /* Grille Techno */
    .tech-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        transition: all 0.2s ease;
    }
    .tech-card:hover {
        background: #1e293b;
        border-color: #38bdf8;
    }

    /* Invitation Navigation / CTA */
    .welcome-cta-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 14px;
        padding: 20px 24px;
        margin-top: 24px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    .presentation-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    .signature-container {
        text-align: center;
        margin-top: 50px;
        margin-bottom: 30px;
        padding-top: 25px;
        border-top: 1px solid #334155;
    }
    .signature-label {
        font-size: 14px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: -5px;
    }
    .signature-text {
        font-family: 'Great Vibes', cursive;
        font-size: 46px;
        color: #38bdf8;
        letter-spacing: 1.5px;
        text-shadow: 0 0 25px rgba(56, 189, 248, 0.4);
    }
</style>
""")


# ── Chargement des données sans limite artificielle ──────
@st.cache_data(ttl=2)
def load_data():
    session = SessionLocal()
    try:
        posts = session.query(Post).order_by(Post.created_utc.desc()).all()
        data = [p.to_dict() for p in posts]
        df = pd.DataFrame(data)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["created_utc"], format="mixed", errors="coerce")
            df = df.dropna(subset=["timestamp"])
        alerts = session.query(CrisisAlert).order_by(CrisisAlert.created_at.desc()).limit(10).all()
        alerts_data = [a.to_dict() for a in alerts]
        return df, alerts_data
    finally:
        session.close()


init_db()
df, alerts_list = load_data()


# ── Barre Latérale (Navigation & Moniteur Infra) ──────────
with st.sidebar:
    st.html("""
    <div style="text-align: center; padding: 6px 0 16px 0;">
        <img src="https://img.icons8.com/isometric/100/brain.png" width="75" style="display: block; margin: 0 auto 10px auto; filter: drop-shadow(0 4px 10px rgba(56, 189, 248, 0.25));" />
        <div style="font-size: 21px; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;">Sentiment Analysis</div>
        <div style="font-size: 11px; font-weight: 600; color: #64748b; letter-spacing: 0.8px; text-transform: uppercase; margin-top: 4px;">Plateforme NLP / LLM Temps Réel</div>
    </div>
    """)

    navigation = st.radio(
        "nav_menu", 
        ["Présentation", "Dashboard Live", "Recherche Thématique"], 
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # ── Statut Infrastructure Moderne (Zéro puce, zéro émoji) ─
    redis_status_badge = (
        '<span class="infra-badge-online">Connecté</span>'
        if broker.is_connected
        else '<span class="infra-badge-offline">Mémoire</span>'
    )
    stream_count = broker.get_stream_length()
    db_count = f"{len(df):,}"

    infra_html = f"""<div class="infra-card">
<div class="infra-header">
<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="8" x="2" y="2" rx="2" ry="2"/><rect width="20" height="8" x="2" y="14" rx="2" ry="2"/><line x1="6" x2="6.01" y1="6" y2="6"/><line x1="6" x2="6.01" y1="18" y2="18"/></svg>
<span>Statut Infrastructure</span>
</div>
<div class="infra-row">
<div class="infra-label">
<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>
<span>Redis Cloud</span>
</div>
{redis_status_badge}
</div>
<div class="infra-row">
<div class="infra-label">
<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
<span>Stream</span>
</div>
<span class="infra-badge-val">{stream_count} msgs</span>
</div>
<div class="infra-row">
<div class="infra-label">
<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
<span>Base Locale</span>
</div>
<span class="infra-badge-val">SQLite ({db_count})</span>
</div>
<div class="infra-row">
<div class="infra-label">
<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="m9 8 6 4-6 4Z"/></svg>
<span>Modèle NLP</span>
</div>
<span class="infra-badge-model">RoBERTa</span>
</div>
<div class="infra-row">
<div class="infra-label">
<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
<span>Moteur LLM</span>
</div>
<span class="infra-badge-llm">Groq 120B</span>
</div>
</div>"""

    st.html(infra_html)


    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    auto_refresh = st.checkbox("Rafraîchissement automatique (10s)", value=False)
    if auto_refresh:
        time.sleep(10)
        st.cache_data.clear()
        st.rerun()


# =========================================================
# PAGE 1 : PRÉSENTATION DU PROJET (Chaleureuse & Pro)
# =========================================================
if navigation == "Présentation":
    # ── En-tête Hero Lumineux & Chaleureux ────────────────
    st.html("""
    <div class="presentation-hero">
        <img src="https://img.icons8.com/isometric/120/brain.png" width="85" style="display: block; margin: 0 auto 10px auto; filter: drop-shadow(0 8px 18px rgba(56, 189, 248, 0.3));" />
        
        <div class="hero-badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
            <span>Veille Stratégique & IA Émotionnelle en Temps Réel</span>
        </div>
        
        <div class="presentation-title">Sentiment Analysis</div>
        <div class="presentation-subtitle">
            Une plateforme décisionnelle conçue pour capter le pouls des opinions numériques, 
            anticiper les signaux faibles et transformer les flux massifs de discussions en diagnostics exécutifs clairs et actionnables.
        </div>
        
        <div class="hero-stats-grid">
            <div class="hero-stat-card">
                <div class="hero-stat-val">&lt; 100 ms</div>
                <div class="hero-stat-lbl">Inférence RoBERTa</div>
            </div>
            <div class="hero-stat-card">
                <div class="hero-stat-val">120 Mrd</div>
                <div class="hero-stat-lbl">Modèle Groq LLM</div>
            </div>
            <div class="hero-stat-card">
                <div class="hero-stat-val">Multi-Sources</div>
                <div class="hero-stat-lbl">Reddit, Presse & Flux</div>
            </div>
            <div class="hero-stat-card">
                <div class="hero-stat-val">100% Temps Réel</div>
                <div class="hero-stat-lbl">Redis Event Stream</div>
            </div>
        </div>
    </div>
    """)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # ── Section 1 : La Vision — Les 3 Piliers de Valeur ───
    st.markdown("""
    <div style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #38bdf8; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h5"/><path d="M17 12h5"/><path d="M12 2v5"/><path d="M12 17v5"/><circle cx="12" cy="12" r="7"/></svg>
        <span>Ce que la Plateforme Apporte aux Décideurs</span>
    </div>
    """, unsafe_allow_html=True)

    col_pil1, col_pil2, col_pil3 = st.columns(3)

    with col_pil1:
        st.html("""
        <div class="pillar-card">
            <div class="pillar-icon-box" style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.35);">
                <img src="https://img.icons8.com/isometric/96/headphones.png" width="28" height="28" style="display: block; margin: auto;" />
            </div>
            <div class="pillar-title">Écoute Active & Impartiale</div>
            <div class="pillar-desc">
                Capter la voix brute du public sur les forums communautaires et les fils d'actualités mondiaux, 
                sans filtre ni délai d'attente, pour saisir immédiatement les ressentis spontanés.
            </div>
        </div>
        """)

    with col_pil2:
        st.html("""
        <div class="pillar-card">
            <div class="pillar-icon-box" style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.35);">
                <img src="https://img.icons8.com/fluency/96/radar.png" width="26" height="26" style="display: block; margin: auto;" />
            </div>
            <div class="pillar-title">Vigilance & Détection Précoce</div>
            <div class="pillar-desc">
                Dépasser la simple volumétrie : évaluer la polarité continue et surveiller les anomalies 
                statistiques (Z-Score) pour désamorcer les crises réputationnelles avant leur emballement.
            </div>
        </div>
        """)

    with col_pil3:
        st.html("""
        <div class="pillar-card">
            <div class="pillar-icon-box" style="background: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.35);">
                <img src="https://img.icons8.com/fluency/96/artificial-intelligence.png" width="26" height="26" style="display: block; margin: auto;" />
            </div>
            <div class="pillar-title">Décisions Éclairées par l'IA</div>
            <div class="pillar-desc">
                Transformer les milliers de publications en Flash Reports narratifs structurés : 
                ce que les gens aiment, ce qu'ils détestent, les controverses clés et des recommandations concrètes.
            </div>
        </div>
        """)

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # ── Section 2 : Parcours de la Donnée (Pipeline de Bout en Bout) ──
    st.markdown("""
    <div style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #38bdf8; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14.5A3.5 3.5 0 0 0 7.5 18H18a4 4 0 0 0 4-4 4 4 0 0 0-3.5-3.9 7 7 0 0 0-13.8 1.4A3.5 3.5 0 0 0 4 14.5Z"/></svg>
        <span>Le Voyage de l'Information : Du Signal Brut au Diagnostic</span>
    </div>
    """, unsafe_allow_html=True)

    col_pipe1, col_pipe2, col_pipe3, col_pipe4 = st.columns(4)

    with col_pipe1:
        st.html("""
        <div class="pipeline-step-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <img src="https://img.icons8.com/fluency/48/rss.png" width="20" height="20" />
                <span style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; font-weight: 800; font-size: 11px; padding: 2px 7px; border-radius: 6px;">ÉTAPE 01</span>
            </div>
            <div style="font-weight: 700; font-size: 15px; color: #f8fafc; margin-bottom: 6px;">Captation & Flux</div>
            <div style="font-size: 13px; color: #94a3b8; line-height: 1.5;">
                Ingestion multi-canale (Reddit PRAW, Google News RSS et streaming synthétique) en quasi temps réel.
            </div>
        </div>
        """)

    with col_pipe2:
        st.html("""
        <div class="pipeline-step-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/redis/redis-original.svg" width="20" height="20" />
                <span style="background: rgba(239, 68, 68, 0.2); color: #f87171; font-weight: 800; font-size: 11px; padding: 2px 7px; border-radius: 6px;">ÉTAPE 02</span>
            </div>
            <div style="font-weight: 700; font-size: 15px; color: #f8fafc; margin-bottom: 6px;">Broker Événementiel</div>
            <div style="font-size: 13px; color: #94a3b8; line-height: 1.5;">
                File de messages distribuée via Redis Streams Cloud : tampon élastique sans perte de paquets.
            </div>
        </div>
        """)

    with col_pipe3:
        st.html("""
        <div class="pipeline-step-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <img src="https://cdn.simpleicons.org/huggingface/FFD21E" width="20" height="20" />
                <span style="background: rgba(34, 197, 94, 0.2); color: #4ade80; font-weight: 800; font-size: 11px; padding: 2px 7px; border-radius: 6px;">ÉTAPE 03</span>
            </div>
            <div style="font-weight: 700; font-size: 15px; color: #f8fafc; margin-bottom: 6px;">Inférence RoBERTa</div>
            <div style="font-size: 13px; color: #94a3b8; line-height: 1.5;">
                Deep Learning NLP (Cardiff NLP) pour quantifier la polarité [-1.0, +1.0] et le niveau de confiance.
            </div>
        </div>
        """)

    # Récupérer l'URI de l'icône officielle Groq
    groq_icon_uri = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Groq_logo.svg/1280px-Groq_logo.svg.png"
    try:
        with open("dashboard/static/groq_data_uri.txt", "r") as gf:
            groq_icon_uri = gf.read().strip()
    except Exception:
        pass

    with col_pipe4:
        st.html(f"""
        <div class="pipeline-step-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <img src="{groq_icon_uri}" width="42" height="18" style="object-fit: contain;" />
                <span style="background: rgba(168, 85, 247, 0.2); color: #c084fc; font-weight: 800; font-size: 11px; padding: 2px 7px; border-radius: 6px;">ÉTAPE 04</span>
            </div>
            <div style="font-weight: 700; font-size: 15px; color: #f8fafc; margin-bottom: 6px;">Brief Décisionnel</div>
            <div style="font-size: 13px; color: #94a3b8; line-height: 1.5;">
                Synthèses d'intelligence exécutive par Groq (GPT-OSS 120B) livrées en moins de 2 secondes.
            </div>
        </div>
        """)

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # ── Section 3 : Stack Technique Moderne (Vraies Icônes) ────
    st.markdown("""
    <div style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #38bdf8; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="m9 8 6 4-6 4Z"/></svg>
        <span>Technologies Éprouvées au Cœur du Système</span>
    </div>
    """, unsafe_allow_html=True)

    t_col1, t_col2, t_col3 = st.columns(3)

    with t_col1:
        st.html("""
        <div class="tech-card">
            <div style="display: flex; align-items: center; gap: 6px; flex-shrink: 0;">
                <img src="https://img.icons8.com/color/96/reddit.png" width="30" height="30" />
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/google/google-original.svg" width="24" height="24" />
            </div>
            <div>
                <div style="font-weight: 700; font-size: 14px; color: #f8fafc;">Reddit API & Google News</div>
                <div style="font-size: 12px; color: #94a3b8;">Captation continue d'opinions et d'actualités</div>
            </div>
        </div>
        """)
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.html("""
        <div class="tech-card">
            <div style="flex-shrink: 0;">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/redis/redis-original.svg" width="32" height="32" />
            </div>
            <div>
                <div style="font-weight: 700; font-size: 14px; color: #f8fafc;">Redis Streams Cloud</div>
                <div style="font-size: 12px; color: #94a3b8;">Broker de messages serverless à latence sub-milliseconde</div>
            </div>
        </div>
        """)

    with t_col2:
        st.html("""
        <div class="tech-card">
            <div style="flex-shrink: 0;">
                <img src="https://cdn.simpleicons.org/huggingface/FFD21E" width="32" height="32" />
            </div>
            <div>
                <div style="font-weight: 700; font-size: 14px; color: #f8fafc;">Twitter-RoBERTa-Base</div>
                <div style="font-size: 12px; color: #94a3b8;">Modèle Transformers spécialisé dans le langage social</div>
            </div>
        </div>
        """)
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.html(f"""
        <div class="tech-card">
            <div style="flex-shrink: 0; background: rgba(245, 80, 54, 0.1); padding: 4px 8px; border-radius: 8px; border: 1px solid rgba(245, 80, 54, 0.25);">
                <img src="{groq_icon_uri}" width="65" height="24" style="display: block; object-fit: contain;" />
            </div>
            <div>
                <div style="font-weight: 700; font-size: 14px; color: #f8fafc;">Groq Cloud API (120B)</div>
                <div style="font-size: 12px; color: #94a3b8;">Moteur d'inférence LPU pour synthèses instantanées</div>
            </div>
        </div>
        """)

    with t_col3:
        st.html("""
        <div class="tech-card">
            <div style="display: flex; align-items: center; gap: 6px; flex-shrink: 0;">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/sqlite/sqlite-original.svg" width="28" height="28" />
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="26" height="26" />
            </div>
            <div>
                <div style="font-weight: 700; font-size: 14px; color: #f8fafc;">SQLAlchemy & SQLite</div>
                <div style="font-size: 12px; color: #94a3b8;">Stockage relationnel ACID, historique & détection d'anomalies</div>
            </div>
        </div>
        """)
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.html("""
        <div class="tech-card">
            <div style="display: flex; align-items: center; gap: 6px; flex-shrink: 0;">
                <img src="https://cdn.simpleicons.org/streamlit/FF4B4B" width="30" height="30" />
                <img src="https://cdn.simpleicons.org/plotly/3F4F75" width="26" height="26" />
            </div>
            <div>
                <div style="font-weight: 700; font-size: 14px; color: #f8fafc;">Streamlit & Plotly</div>
                <div style="font-size: 12px; color: #94a3b8;">Visualisation interactive, réactivité & filtres temps réel</div>
            </div>
        </div>
        """)

    # ── Section 4 : Invitation à l'Action Chaleureuse ─────
    st.html("""
    <div class="welcome-cta-box">
        <div style="font-size: 20px; font-weight: 800; color: #f8fafc; margin-bottom: 6px;">
            Prêt à explorer les signaux d'opinion ?
        </div>
        <div style="font-size: 14px; color: #94a3b8; max-width: 680px; margin: 0 auto 16px auto; line-height: 1.6;">
            Accédez directement à vos espaces de travail depuis le menu latéral à gauche :
        </div>
        <div style="display: flex; justify-content: center; gap: 16px; flex-wrap: wrap;">
            <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 8px 16px; font-size: 13px; color: #38bdf8; display: inline-flex; align-items: center; gap: 8px;">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                <span><strong>Dashboard Live</strong> : Flux en direct & déclencheurs d'ingestion</span>
            </div>
            <div style="background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 8px; padding: 8px 16px; font-size: 13px; color: #c084fc; display: inline-flex; align-items: center; gap: 8px;">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#c084fc" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                <span><strong>Recherche Thématique</strong> : Audit sur-mesure & Dossier IA</span>
            </div>
        </div>
    </div>
    """)

    # ── Signature Calligraphique Finale ───────────────────
    st.html("""
    <div class="signature-container">
        <div style="font-size: 14px; color: #94a3b8; font-style: italic; margin-bottom: 8px;">
            « Transformer le tumulte des données en clarté stratégique. »
        </div>
        <div class="signature-label">Conception & Réalisation</div>
        <div class="signature-text">Fait par Jeff OBANDA</div>
    </div>
    """)


# =========================================================
# PAGE 2 : DASHBOARD LIVE
# =========================================================
elif navigation == "Dashboard Live":
    st.title("Real-Time Sentiment Intelligence Platform")
    st.markdown("Surveillance de réputation en direct, détection statistique d'anomalies et Flash Reports exécutifs par LLM.")

    # ── Panneau Central des 3 Boutons d'Action Directe ───────
    st.markdown("""
    <div style="background: #1e293b; border-left: 4px solid #38bdf8; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; display: flex; align-items: center; gap: 12px;">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0;"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
        <div>
            <strong style="color: #38bdf8;">DÉCLENCHEURS D'INGESTION EN DIRECT :</strong>
            <span style="color: #94a3b8; font-size: 14px;"> Cliquez sur un bouton pour collecter de nouvelles données. Le compteur de posts et tous les KPIs s'actualisent instantanément !</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_act1, col_act2, col_act3 = st.columns(3)

    with col_act1:
        if st.button("Collecter Reddit en Direct", use_container_width=True, type="primary"):
            with st.spinner("Collecte des posts réels sur Reddit (tech, stocks, gaming)..."):
                count = reddit_collector.collect_and_process(subreddits=["technology", "stocks", "gaming"], limit_per_sub=5)
                st.cache_data.clear()
                st.toast(f"{count} vrais posts Reddit collectés, scorés et stockés !")
                st.rerun()

    with col_act2:
        if st.button("Collecter Actualités en Direct", use_container_width=True, type="primary"):
            with st.spinner("Collecte des actualités mondiales en direct..."):
                count = news_collector.collect_and_process(limit_per_topic=5)
                st.cache_data.clear()
                st.toast(f"{count} articles d'actualité collectés, scorés et stockés !")
                st.rerun()

    with col_act3:
        if st.button("Simuler 5 Posts (Stream)", use_container_width=True):
            with st.spinner("Injection streaming en cours..."):
                for _ in range(5):
                    generate_single_event()
                st.cache_data.clear()
                st.toast("5 événements streamés dans Redis Cloud et SQLite !")
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if df.empty:
        st.warning("La base de données est vide pour l'instant. Utilisez les boutons ci-dessus pour collecter des données !")
        st.stop()

    # ── KPIs Exécutifs Synchronisés Dynamiquement ────────────
    total_posts = len(df)
    pos_ratio = (df["predicted_sentiment"] == "positive").mean() * 100
    neg_ratio = (df["predicted_sentiment"] == "negative").mean() * 100
    neu_ratio = (df["predicted_sentiment"] == "neutral").mean() * 100
    mean_polarity = df["polarity"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                <span>Posts Analysés</span>
            </div>
            <div class="kpi-value" style="color: #38bdf8;">{total_posts:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
                <span>Sentiment Positif</span>
            </div>
            <div class="kpi-value" style="color: #22c55e;">{pos_ratio:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>
                <span>Sentiment Négatif</span>
            </div>
            <div class="kpi-value" style="color: #ef4444;">{neg_ratio:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        color_p = "#22c55e" if mean_polarity >= 0 else "#ef4444"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{color_p}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                <span>Indice Polarité</span>
            </div>
            <div class="kpi-value" style="color: {color_p};">{mean_polarity:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        alert_color = "#ef4444" if len(alerts_list) > 0 else "#22c55e"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{alert_color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                <span>Alertes Crise</span>
            </div>
            <div class="kpi-value" style="color: {alert_color};">{len(alerts_list)} Active(s)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Alerte de Crise & Flash Report LLM ────────────────────
    if neg_ratio > 35.0:
        st.html(f"""
        <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 12px 16px; display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            <span style="color: #f87171; font-weight: 700;">ALERTE CRISE CRITIQUE DÉTECTÉE : Taux de négativité anormalement élevé ({neg_ratio:.1f}% &gt; seuil 35%).</span>
        </div>
        """)

    # ── 1. Graphique Temporel de la Réputation ────────────────
    st.html("""
    <div style="font-size: 20px; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 10px; margin: 20px 0 10px 0;">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
        <span>Dynamique Temporelle de la Réputation</span>
    </div>
    """)
    
    df_time = df.set_index("timestamp").resample("1D").agg(
        count=("id", "count"),
        polarity=("polarity", "mean"),
        negative_ratio=("predicted_sentiment", lambda s: (s == "negative").mean() * 100)
    ).dropna().reset_index()

    if len(df_time) > 1:
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=df_time["timestamp"], y=df_time["polarity"],
            mode="lines+markers", name="Polarité Moyenne",
            line=dict(color="#38bdf8", width=2.5)
        ))
        fig_time.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_time.update_layout(
            template="plotly_dark",
            title="Évolution de la Polarité Continue [-1.0, +1.0]",
            yaxis_title="Score de Polarité",
            xaxis_title="Temps",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Données temporelles insuffisantes pour tracer la série. Utilisez les boutons de collecte.")

    # ── Flash Report de Crise LLM (Groq) : Juste en bas de la courbe ──
    with st.expander("Générateur / Historique des Flash Reports de Crise LLM (Groq)", expanded=False):
        col_rep_act, col_rep_view = st.columns([1, 2])
        with col_rep_act:
            st.html("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
                <h4 style="margin: 0; color: #f8fafc; font-size: 16px;">Générer un nouveau rapport</h4>
            </div>
            """)
            st.caption("Déclenche un audit par GPT-OSS 120B / LLaMA sur les derniers posts négatifs.")
            if st.button("Déclencher Flash Report de Crise", use_container_width=True):
                with st.spinner("Génération de la synthèse décisionnelle via Groq..."):
                    neg_posts = df[df["predicted_sentiment"] == "negative"].head(25).to_dict("records")
                    if neg_posts:
                        report = generate_crisis_report(neg_posts, alert_date=datetime.utcnow(), negative_ratio=neg_ratio)
                        st.cache_data.clear()
                        st.success("Rapport généré avec succès !")
                        st.rerun()
                    else:
                        st.info("Aucun post négatif à analyser.")

        with col_rep_view:
            if alerts_list:
                latest_alert = alerts_list[0]
                st.markdown(f"**Dernier rapport enregistré ({latest_alert['alert_date']})** :")
                st.markdown(latest_alert["llm_report"])
            else:
                st.info("Aucun rapport de crise n'a encore été généré. Cliquez sur le bouton pour en créer un.")

    # ── 2. En bas de la courbe : Distribution & Profil Émotionnel du Jour (avec filtre date) ──
    from datetime import date, timedelta
    max_date = df["timestamp"].dt.date.max()
    yesterday_date = max_date - timedelta(days=1)

    # Barre de filtre date (borne inférieure : hier)
    col_filter_date, col_filter_info = st.columns([1, 2])
    with col_filter_date:
        selected_date = st.date_input(
            "Filtrer par date :",
            value=max_date,
            min_value=yesterday_date,
            max_value=max_date,
            help="Sélectionnez Aujourd'hui ou Hier"
        )

    # Filtrer les données selon la date sélectionnée
    df_day = df[df["timestamp"].dt.date == selected_date]
    day_label = "Aujourd'hui" if selected_date == max_date else "Hier"

    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #1e293b, #0f172a); border-left: 4px solid #22c55e; padding: 10px 16px; border-radius: 8px; margin-top: 10px; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>
        <div>
            <span style="font-size: 16px; font-weight: 700; color: #f8fafc;">Distribution & Profil Émotionnel par Sujet</span>
            <span style="color: #22c55e; font-weight: 600; font-size: 15px;"> &bull; {day_label} : {selected_date.strftime('%d %B %Y')}</span>
            <span style="color: #94a3b8; font-size: 13px;"> ({len(df_day):,} posts analysés)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not df_day.empty:
        col_day1, col_day2 = st.columns(2)

        with col_day1:
            fig_pie_day = px.pie(
                df_day, 
                names="predicted_sentiment",
                color="predicted_sentiment",
                color_discrete_map={"positive": "#22c55e", "negative": "#ef4444", "neutral": "#3b82f6"},
                title=f"Répartition des Sentiments ({selected_date.strftime('%d/%m/%Y')})",
                hole=0.45,
                template="plotly_dark"
            )
            fig_pie_day.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_pie_day, use_container_width=True)

        with col_day2:
            topic_cross_day = pd.crosstab(df_day["topic"], df_day["predicted_sentiment"], normalize="index") * 100
            fig_bar_day = px.bar(
                topic_cross_day,
                barmode="stack",
                color_discrete_map={"positive": "#22c55e", "negative": "#ef4444", "neutral": "#3b82f6"},
                title=f"Profil Émotionnel par Sujet ({selected_date.strftime('%d/%m/%Y')})",
                template="plotly_dark"
            )
            fig_bar_day.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20), yaxis_title="% du sujet")
            st.plotly_chart(fig_bar_day, use_container_width=True)
    else:
        st.info(f"Aucun post enregistré pour le {selected_date.strftime('%d/%m/%Y')}.")

    # ── 3. Flux des Posts Filtrés par la Date (Pagination par 5) ──
    with st.expander(f"Flux des Posts ({selected_date.strftime('%d/%m/%Y')} — {len(df_day)} posts)", expanded=True):
        col_filt, col_nav = st.columns([2.2, 1])

        with col_filt:
            filter_sent_sel = st.segmented_control(
                "Filtrer par sentiment :", 
                options=["Tous", "Positif", "Négatif", "Neutre"], 
                default="Tous",
                key="feed_sent_filter"
            )
            sent_map = {"Positif": "positive", "Négatif": "negative", "Neutre": "neutral"}
            filter_sent = sent_map.get(filter_sent_sel or "Tous", "Tous")

        # Filtrage par sentiment
        feed_posts = df_day.copy()
        if filter_sent != "Tous":
            feed_posts = feed_posts[feed_posts["predicted_sentiment"] == filter_sent]

        total_posts_day = len(feed_posts)

        if total_posts_day == 0:
            st.info(f"Aucun post ne correspond aux critères pour le {selected_date.strftime('%d/%m/%Y')}.")
        else:
            # Paramètres de pagination (5 posts par page)
            posts_per_page = 5
            total_pages = max(1, (total_posts_day + posts_per_page - 1) // posts_per_page)

            # Pagination moderne épurée avec boutons chevrons
            if "feed_page_idx" not in st.session_state:
                st.session_state["feed_page_idx"] = 1
            
            if st.session_state["feed_page_idx"] > total_pages:
                st.session_state["feed_page_idx"] = total_pages
            if st.session_state["feed_page_idx"] < 1:
                st.session_state["feed_page_idx"] = 1

            with col_nav:
                p_col1, p_col2, p_col3 = st.columns([1, 1.8, 1])
                with p_col1:
                    if st.button("‹ Préc.", disabled=(st.session_state["feed_page_idx"] <= 1), use_container_width=True, key="btn_prev_feed", help="Page précédente"):
                        st.session_state["feed_page_idx"] -= 1
                        st.rerun()
                with p_col2:
                    st.markdown(f"<div style='text-align: center; padding-top: 6px; font-weight: 600; font-size: 13px; color: #94a3b8;'>Page {st.session_state['feed_page_idx']} / {total_pages}</div>", unsafe_allow_html=True)
                with p_col3:
                    if st.button("Suiv. ›", disabled=(st.session_state["feed_page_idx"] >= total_pages), use_container_width=True, key="btn_next_feed", help="Page suivante"):
                        st.session_state["feed_page_idx"] += 1
                        st.rerun()

            current_page = st.session_state["feed_page_idx"]
            start_idx = (current_page - 1) * posts_per_page
            end_idx = min(start_idx + posts_per_page, total_posts_day)

            st.caption(f"Affichage des posts **{start_idx + 1} à {end_idx}** sur un total de **{total_posts_day}** posts ({day_label} {selected_date.strftime('%d/%m/%Y')})")

            # Afficher exactement les 5 posts de la page courante
            page_slice = feed_posts.iloc[start_idx:end_idx]

            for _, row in page_slice.iterrows():
                sent_class = "pos" if row["predicted_sentiment"] == "positive" else "neg" if row["predicted_sentiment"] == "negative" else "neu"
                label_text = "POSITIF" if row["predicted_sentiment"] == "positive" else "NÉGATIF" if row["predicted_sentiment"] == "negative" else "NEUTRE"
                badge_html = f'<span class="badge-pill badge-{sent_class}">{label_text}</span>'
                
                st.markdown(f"""
                <div class="post-card {sent_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; color: #f8fafc;">[{row['source'].upper()}] {row['channel']} &bull; {row['author']}</span>
                        <span>{badge_html} <small style="color: #94a3b8; margin-left: 6px;">({row['sentiment_score']:.2f})</small></span>
                    </div>
                    <div style="color: #cbd5e1; font-size: 15px;">{row['text']}</div>
                    <div style="margin-top: 6px; font-size: 11px; color: #64748b;">Topic: {row['topic'].capitalize()} &bull; Horodatage: {row['created_utc']}</div>
                </div>
                """, unsafe_allow_html=True)

    # Signature calligraphique en pied de page
    st.markdown("""
    <div class="signature-container" style="margin-top: 40px;">
        <div class="signature-label">Conception & Réalisation</div>
        <div class="signature-text" style="font-size: 38px;">Fait par Jeff OBANDA</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE 3 : RECHERCHE & SYNTHÈSE THÉMATIQUE
# =========================================================
elif navigation == "Recherche Thématique":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #3b82f6; border-radius: 14px; padding: 22px 26px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 16px;">
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            <div>
                <h2 style="margin: 0; color: #f8fafc; font-size: 24px;">Recherche & Intelligence d'Opinion Thématique</h2>
                <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px; line-height: 1.5;">
                    Analysez l'opinion publique sur n'importe quel sujet (produit, marque, technologie, personnalité) 
                    sur une période définie (ex : 1 mois). Inférence locale par le modèle <strong>Twitter-RoBERTa</strong> 
                    et dossier décisionnel approfondi généré par le modèle LLM <strong>Groq (GPT-OSS 120B)</strong>.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Formulaire de recherche et configuration des paramètres ──
    with st.container():
        st.html("""
        <div style="font-size: 20px; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 10px; margin: 15px 0 14px 0;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"/>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
            </svg>
            <span>Paramètres de la Requête</span>
        </div>
        """)
        col_q1, col_q2, col_q3 = st.columns([2, 1, 1])

        with col_q1:
            search_query = st.text_input(
                "Sujet, Marque ou Mot-clé :",
                value="ChatGPT",
                placeholder="Ex: ChatGPT, Tesla, iPhone 16, Nvidia RTX, Bitcoin...",
                help="Saisissez le terme ou l'entité à auditer."
            )

        with col_q2:
            period_choice = st.selectbox(
                "Période temporelle :",
                options=[
                    ("30d", "Dernier mois (30 jours)"),
                    ("7d", "Dernière semaine (7 jours)"),
                    ("1d", "Dernières 24 heures")
                ],
                format_func=lambda x: x[1],
                index=0,
                help="Fenêtre temporelle des publications ciblées."
            )

        with col_q3:
            limit_choice = st.slider(
                "Volume max par requête :",
                min_value=10,
                max_value=50,
                value=25,
                step=5,
                help="Contrôle le quota de posts collectés pour respecter les rate limits et la rapidité du modèle local."
            )

        col_opt1, col_opt2 = st.columns([2, 1])
        with col_opt1:
            source_choice = st.selectbox(
                "Source des données :",
                options=[
                    ("all", "Mixte (Presse Web + Communautés Reddit)"),
                    ("news", "Actualités & Presse Web (Google News RSS)"),
                    ("reddit", "Discussions & Avis Réels (Reddit)"),
                    ("twitter", "Échantillonnage Réseaux Sociaux (Tweets)")
                ],
                format_func=lambda x: x[1],
                index=0
            )

        with col_opt2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            save_in_db = st.checkbox("Sauvegarder dans la base locale", value=True)

        st.markdown("<br>", unsafe_allow_html=True)
        btn_col1, _ = st.columns([1, 2])
        with btn_col1:
            launch_search = st.button("Lancer l'Analyse d'Opinion", type="primary", use_container_width=True)

    # ── Traitement de la recherche ──
    if launch_search and search_query.strip():
        with st.status(f"Traitement en cours pour '{search_query}'...", expanded=True) as status:
            st.write(f"1/3. Collecte ciblée sur **{search_query}** ({period_choice[1]}, quota: {limit_choice} posts)...")
            results = search_and_analyze_topic(
                query=search_query.strip(),
                period=period_choice[0],
                source=source_choice[0],
                limit=limit_choice,
                save_to_db=save_in_db
            )
            
            st.write(f"2/3. Nettoyage NLP et classification par **Twitter-RoBERTa** terminée ({len(results['posts'])} posts scorés)...")
            
            st.write("3/3. Rédaction du Dossier d'Intelligence d'Opinion par le modèle LLM **Groq (GPT-OSS 120B)**...")
            report = generate_topic_deep_dive(
                subject=search_query.strip(),
                period_label=period_choice[1],
                posts_data=results["posts"],
                metrics=results["metrics"]
            )

            status.update(label="Recherche et Dossier d'Intelligence générés avec succès !", state="complete", expanded=False)

        # Enregistrer dans le session_state
        st.session_state["topic_query"] = search_query.strip()
        st.session_state["topic_period"] = period_choice[1]
        st.session_state["topic_results"] = results
        st.session_state["topic_report"] = report

    # ── Affichage des Résultats ──
    if "topic_results" in st.session_state and st.session_state["topic_results"]:
        res = st.session_state["topic_results"]
        metrics = res["metrics"]
        posts = res["posts"]
        report = st.session_state.get("topic_report")
        query_saved = st.session_state.get("topic_query", "")
        period_saved = st.session_state.get("topic_period", "")

        st.markdown("---")
        st.html(f"""
        <div style="font-size: 20px; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 10px; margin: 20px 0 15px 0;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
                <line x1="8" y1="21" x2="16" y2="21"/>
                <line x1="12" y1="17" x2="12" y2="21"/>
            </svg>
            <span>Bilan Synthétique : <strong style="color: #38bdf8;">{query_saved}</strong> <span style="font-size: 15px; color: #94a3b8; font-weight: 500;">({period_saved})</span></span>
        </div>
        """)

        # 4 Cartes KPI
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    <span>Posts Analysés</span>
                </div>
                <div class="kpi-value" style="color: #38bdf8;">{metrics['total_posts']}</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Quota respecté</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            pol = metrics['avg_polarity']
            pol_color = "#22c55e" if pol >= 0.1 else "#ef4444" if pol <= -0.1 else "#38bdf8"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{pol_color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                    <span>Polarité Nette</span>
                </div>
                <div class="kpi-value" style="color: {pol_color};">{pol:+.2f}</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Échelle [-1.0, +1.0]</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
                    <span>Taux d'Adhésion</span>
                </div>
                <div class="kpi-value" style="color: #22c55e;">{metrics['pos_ratio']:.1f}%</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 4px;">{metrics['pos_count']} posts favorables</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>
                    <span>Taux de Rejet</span>
                </div>
                <div class="kpi-value" style="color: #ef4444;">{metrics['neg_ratio']:.1f}%</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 4px;">{metrics['neg_count']} posts critiques</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Graphiques : Jauge de Polarité & Donut ──
        col_g1, col_g2 = st.columns([1, 1])

        with col_g1:
            # Jauge de Polarité
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics['avg_polarity'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Thermomètre de Polarité ({query_saved})", 'font': {'size': 18, 'color': '#f8fafc'}},
                number={'valueformat': "+.2f", 'font': {'size': 32, 'color': pol_color}},
                gauge={
                    'axis': {'range': [-1.0, 1.0], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': pol_color},
                    'bgcolor': "#1e293b",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [-1.0, -0.2], 'color': 'rgba(239, 68, 68, 0.3)'},
                        {'range': [-0.2, 0.2], 'color': 'rgba(59, 130, 246, 0.25)'},
                        {'range': [0.2, 1.0], 'color': 'rgba(34, 197, 94, 0.3)'}
                    ],
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#f8fafc', 'family': 'Inter'},
                height=260,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_g2:
            # Donut répartition
            donut_labels = ["Positif", "Neutre", "Négatif"]
            donut_values = [metrics['pos_count'], metrics['neu_count'], metrics['neg_count']]
            donut_colors = ["#22c55e", "#3b82f6", "#ef4444"]

            fig_donut = go.Figure(data=[go.Pie(
                labels=donut_labels,
                values=donut_values,
                hole=.55,
                marker=dict(colors=donut_colors),
                textinfo='label+percent',
                textfont=dict(size=13, color="#f8fafc"),
            )])
            fig_donut.update_layout(
                title=dict(text="Répartition Émotionnelle des Signaux", font=dict(size=18, color='#f8fafc')),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                height=260,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # ── Rapport Exécutif Groq (Dossier d'Intelligence) ──
        if report:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #0f172a, #1e293b); border: 1px solid #38bdf8; border-radius: 14px; padding: 26px; margin-top: 15px; margin-bottom: 25px; box-shadow: 0 10px 25px -5px rgba(56, 189, 248, 0.2);">
                <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 18px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
                        <strong style="font-size: 17px; color: #38bdf8; text-transform: uppercase; letter-spacing: 1px;">Dossier d'Intelligence d'Opinion Approfondi</strong>
                    </div>
                    <span style="background: #0284c7; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">Groq GPT-OSS 120B</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(report)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("La synthèse Groq n'a pas pu être générée. Vérifiez votre clé GROQ_API_KEY dans le fichier .env.")

        # ── Explorateur des Posts / Tweets Collectés ──
        with st.expander(f"Explorer les {len(posts)} publications & verbatims analysés", expanded=False):
            col_filt_sub, _ = st.columns([2.5, 1])
            with col_filt_sub:
                filter_view_sel = st.segmented_control(
                    "Filtrer l'échantillon par sentiment :",
                    options=["Tous", "Positif", "Négatif", "Neutre"],
                    default="Tous",
                    key="topic_filter_view"
                )
                sent_map = {"Positif": "positive", "Négatif": "negative", "Neutre": "neutral"}
                filter_view = sent_map.get(filter_view_sel or "Tous", "Tous")

            filtered_posts = posts if filter_view == "Tous" else [p for p in posts if p["predicted_sentiment"] == filter_view]

            for p in filtered_posts:
                s_class = "pos" if p["predicted_sentiment"] == "positive" else "neg" if p["predicted_sentiment"] == "negative" else "neu"
                label_text = "POSITIF" if p["predicted_sentiment"] == "positive" else "NÉGATIF" if p["predicted_sentiment"] == "negative" else "NEUTRE"
                badge_html = f'<span class="badge-pill badge-{s_class}">{label_text}</span>'
                
                st.markdown(f"""
                <div class="post-card {s_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; color: #f8fafc;">[{p['source'].upper()}] {p['channel']} &bull; {p['author']}</span>
                        <span>{badge_html} <small style="color: #94a3b8; margin-left: 6px;">(Score: {p['sentiment_score']:.2f} | Pol: {p['polarity']:+.2f})</small></span>
                    </div>
                    <div style="color: #cbd5e1; font-size: 14px; line-height: 1.5;">{p['text']}</div>
                    <div style="margin-top: 6px; font-size: 11px; color: #64748b;">Horodatage : {p['created_utc']}</div>
                </div>
                """, unsafe_allow_html=True)

    # Signature calligraphique en pied de page
    st.markdown("""
    <div class="signature-container" style="margin-top: 40px;">
        <div class="signature-label">Conception & Réalisation</div>
        <div class="signature-text" style="font-size: 38px;">Fait par Jeff OBANDA</div>
    </div>
    """, unsafe_allow_html=True)

