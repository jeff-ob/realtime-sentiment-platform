"""
database.py — Modèles et persistance SQLite / SQLAlchemy
Real-Time Sentiment Intelligence Platform
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, desc
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

# Ajouter la racine du projet au PYTHONPATH pour importer config
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# Configurer stdout en UTF-8 pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

Base = declarative_base()


class Post(Base):
    """Table des publications analysées en temps réel."""
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(String(100), unique=True, index=True)
    source = Column(String(50), default="simulator", index=True)      # reddit, newsapi, simulator
    channel = Column(String(100), default="general", index=True)      # ex: r/technology, r/stocks
    author = Column(String(100))
    text_raw = Column(Text, nullable=False)
    text_clean = Column(Text, nullable=False)
    topic = Column(String(50), index=True)                            # tech, finance, gaming, etc.
    topic_id = Column(Integer, default=-1)                            # ID BERTopic
    topic_label = Column(String(100))                                 # Label BERTopic
    predicted_sentiment = Column(String(20), index=True)              # positive, neutral, negative
    sentiment_score = Column(Float, default=0.0)                      # Score de confiance Softmax [0, 1]
    polarity = Column(Float, default=0.0, index=True)                 # Polarité continue [-1, 1]
    score = Column(Integer, default=0)                                # Score / Upvotes
    num_comments = Column(Integer, default=0)
    created_utc = Column(DateTime, default=datetime.utcnow, index=True)
    processed_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "post_id": self.post_id,
            "source": self.source,
            "channel": self.channel,
            "author": self.author,
            "text": self.text_raw,
            "text_clean": self.text_clean,
            "topic": self.topic,
            "topic_id": self.topic_id,
            "topic_label": self.topic_label,
            "predicted_sentiment": self.predicted_sentiment,
            "sentiment_score": self.sentiment_score,
            "polarity": self.polarity,
            "score": self.score,
            "num_comments": self.num_comments,
            "created_utc": self.created_utc.isoformat() if self.created_utc else None,
        }


class CrisisAlert(Base):
    """Table des alertes de crise détectées et synthèses LLM."""
    __tablename__ = "crisis_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_date = Column(DateTime, default=datetime.utcnow, index=True)
    negative_ratio = Column(Float, nullable=False)                    # Taux de négativité lors du pic
    post_count = Column(Integer, nullable=False)
    status = Column(String(20), default="ACTIVE")                     # ACTIVE, RESOLVED
    llm_report = Column(Text)                                         # Flash report généré par Groq
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "alert_date": self.alert_date.strftime("%Y-%m-%d"),
            "negative_ratio": self.negative_ratio,
            "post_count": self.post_count,
            "status": self.status,
            "llm_report": self.llm_report,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ── Configuration Moteur SQLite ──────────────────────────
# Assurer l'existence du dossier data
config.DATA_RAW.mkdir(parents=True, exist_ok=True)
config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{config.DATABASE_PATH}",
    echo=False,
    connect_args={"check_same_thread": False}
)

SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


def init_db():
    """Initialise le schéma de la base de données."""
    Base.metadata.create_all(bind=engine)
    print(f"✅ Base de données initialisée : {config.DATABASE_PATH}")


def save_post(data: Dict[str, Any]) -> Optional[Post]:
    """Sauvegarde un post unitaire dans la base."""
    session = SessionLocal()
    try:
        # Éviter les doublons sur post_id
        existing = session.query(Post).filter_by(post_id=data.get("post_id")).first()
        if existing:
            return existing

        created_utc = data.get("created_utc")
        if isinstance(created_utc, str):
            try:
                created_utc = datetime.fromisoformat(created_utc)
            except ValueError:
                created_utc = datetime.utcnow()

        post = Post(
            post_id=data.get("post_id", f"post_{datetime.utcnow().timestamp()}"),
            source=data.get("source", "simulator"),
            channel=data.get("channel", "general"),
            author=data.get("author", "anonymous"),
            text_raw=data.get("text", ""),
            text_clean=data.get("text_clean", ""),
            topic=data.get("topic", "general"),
            topic_id=data.get("topic_id", -1),
            topic_label=data.get("topic_label", "Unassigned"),
            predicted_sentiment=data.get("predicted_sentiment", "neutral"),
            sentiment_score=data.get("sentiment_score", 0.0),
            polarity=data.get("polarity", 0.0),
            score=data.get("score", 0),
            num_comments=data.get("num_comments", 0),
            created_utc=created_utc,
        )
        session.add(post)
        session.commit()
        session.refresh(post)
        return post
    except Exception as e:
        session.rollback()
        print(f"❌ Erreur lors de l'insertion : {e}")
        return None
    finally:
        session.close()


def save_posts_bulk(posts_data: List[Dict[str, Any]]) -> int:
    """Insère une liste de posts par batch."""
    session = SessionLocal()
    count = 0
    try:
        for data in posts_data:
            existing = session.query(Post).filter_by(post_id=data.get("post_id")).first()
            if existing:
                continue

            created_utc = data.get("created_utc")
            if isinstance(created_utc, str):
                try:
                    created_utc = datetime.fromisoformat(created_utc)
                except ValueError:
                    created_utc = datetime.utcnow()

            post = Post(
                post_id=data.get("post_id", f"post_{datetime.utcnow().timestamp()}"),
                source=data.get("source", "simulator"),
                channel=data.get("channel", "general"),
                author=data.get("author", "anonymous"),
                text_raw=data.get("text", ""),
                text_clean=data.get("text_clean", ""),
                topic=data.get("topic", "general"),
                topic_id=data.get("topic_id", -1),
                topic_label=data.get("topic_label", "Unassigned"),
                predicted_sentiment=data.get("predicted_sentiment", "neutral"),
                sentiment_score=data.get("sentiment_score", 0.0),
                polarity=data.get("polarity", 0.0),
                score=data.get("score", 0),
                num_comments=data.get("num_comments", 0),
                created_utc=created_utc,
            )
            session.add(post)
            count += 1

        session.commit()
        return count
    except Exception as e:
        session.rollback()
        print(f"❌ Erreur bulk insert : {e}")
        return 0
    finally:
        session.close()


def get_recent_posts(limit: int = 50) -> List[Dict[str, Any]]:
    """Récupère les posts les plus récents."""
    session = SessionLocal()
    try:
        posts = session.query(Post).order_by(desc(Post.created_utc)).limit(limit).all()
        return [p.to_dict() for p in posts]
    finally:
        session.close()


def save_crisis_alert(alert_date: datetime, negative_ratio: float, post_count: int, llm_report: str) -> CrisisAlert:
    """Enregistre une alerte de crise et son rapport exécutif."""
    session = SessionLocal()
    try:
        alert = CrisisAlert(
            alert_date=alert_date,
            negative_ratio=negative_ratio,
            post_count=post_count,
            llm_report=llm_report,
            status="ACTIVE"
        )
        session.add(alert)
        session.commit()
        session.refresh(alert)
        return alert
    finally:
        session.close()


def get_active_alerts() -> List[Dict[str, Any]]:
    """Récupère les alertes de crise actives."""
    session = SessionLocal()
    try:
        alerts = session.query(CrisisAlert).order_by(desc(CrisisAlert.created_at)).all()
        return [a.to_dict() for a in alerts]
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
