"""
sentiment.py — Analyseur de sentiment RoBERTa
Real-Time Sentiment Intelligence Platform
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Union

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from transformers import pipeline


class SentimentAnalyzer:
    """Singleton pour l'analyse de sentiment avec twitter-roberta."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
            cls._instance.pipeline = None
            cls._instance.polarity_weights = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        return cls._instance

    def _ensure_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline

        print("⏳ Chargement du modèle de sentiment RoBERTa...")
        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=config.SENTIMENT_MODEL,
                tokenizer=config.SENTIMENT_MODEL,
                device=-1,          # CPU
                truncation=True,
                max_length=512,
                local_files_only=True,
            )
        except Exception:
            print("🌐 Modèle non trouvé en cache local. Téléchargement depuis HuggingFace...")
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=config.SENTIMENT_MODEL,
                tokenizer=config.SENTIMENT_MODEL,
                device=-1,          # CPU
                truncation=True,
                max_length=512,
                local_files_only=False,
            )
        print("✅ Modèle RoBERTa prêt pour l'inférence !")
        return self.pipeline

    def analyze_one(self, text: str) -> Dict[str, Any]:
        """Analyse un texte unique et retourne sentiment, score et polarité."""
        if not text or not text.strip():
            return {
                "predicted_sentiment": "neutral",
                "sentiment_score": 0.5,
                "polarity": 0.0
            }

        pipe = self._ensure_pipeline()
        res = pipe([text])[0]
        label = res["label"]
        score = round(res["score"], 4)
        polarity = round(self.polarity_weights.get(label, 0.0) * score, 4)

        return {
            "predicted_sentiment": label,
            "sentiment_score": score,
            "polarity": polarity
        }

    def analyze_batch(self, texts: List[str], batch_size: int = 64) -> List[Dict[str, Any]]:
        """Analyse une liste de textes par batch."""
        if not texts:
            return []

        pipe = self._ensure_pipeline()
        # Remplacer les textes vides
        safe_texts = [t if t.strip() else "neutral text" for t in texts]
        results = []

        for i in range(0, len(safe_texts), batch_size):
            batch = safe_texts[i:i + batch_size]
            preds = pipe(batch)
            for res in preds:
                label = res["label"]
                score = round(res["score"], 4)
                polarity = round(self.polarity_weights.get(label, 0.0) * score, 4)
                results.append({
                    "predicted_sentiment": label,
                    "sentiment_score": score,
                    "polarity": polarity
                })

        return results


# Instance globale réutilisable
analyzer = SentimentAnalyzer()


if __name__ == "__main__":
    samples = [
        "This new software release is incredible! Best tool ever.",
        "The server crashed 5 times today, completely broken.",
        "Here are the release notes for version 2.0."
    ]
    for s in samples:
        res = analyzer.analyze_one(s)
        print(f"\n\"{s}\"")
        print(f"   → Sentiment: {res['predicted_sentiment']} | Score: {res['sentiment_score']} | Polarité: {res['polarity']}")
