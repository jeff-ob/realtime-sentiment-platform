"""
news_collector.py — Collecteur d'actualités réelles (NewsAPI + Fallback Google News RSS)
Real-Time Sentiment Intelligence Platform
"""

import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.nlp.cleaner import clean_text
from src.nlp.sentiment import analyzer
from src.streaming.redis_stream import broker
from src.storage.database import save_post

RSS_FEEDS = {
    "tech": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en",
    "finance": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
}


class NewsCollector:
    """Collecteur d'actualités en continu."""

    def __init__(self):
        self.api_key = config.NEWSAPI_KEY
        self.newsapi = None
        self._init_client()

    def _init_client(self):
        if self.api_key and self.api_key != "your_newsapi_key_here":
            try:
                from newsapi import NewsApiClient
                self.newsapi = NewsApiClient(api_key=self.api_key)
                print("✅ Connecté à NewsAPI officielle.")
            except Exception as e:
                print(f"⚠️  Échec NewsAPI ({e}) → Mode RSS actif.")
        else:
            print("ℹ️  Aucune clé NewsAPI configurée → Mode d'ingestion Google News RSS actif.")

    def fetch_rss_news(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Récupère les actualités fraîches via flux RSS sécurisé."""
        url = RSS_FEEDS.get(topic, RSS_FEEDS["tech"])
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        articles = []
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
                root = ET.fromstring(xml_data)
                items = root.findall(".//item")
                for item in items[:limit]:
                    title_elem = item.find("title")
                    title = title_elem.text if title_elem is not None and title_elem.text else ""
                    source_elem = item.find("source")
                    source_name = source_elem.text if source_elem is not None and source_elem.text else "Google News"
                    link_elem = item.find("link")
                    article_id = link_elem.text if link_elem is not None and link_elem.text else str(time.time())

                    if len(title) < 15:
                        continue

                    articles.append({
                        "post_id": f"news_{abs(hash(article_id))}",
                        "source": "news",
                        "channel": source_name,
                        "author": source_name,
                        "text": title,
                        "topic": topic,
                        "score": 50,
                        "num_comments": 0,
                        "created_utc": datetime.now(timezone.utc).isoformat()
                    })
        except Exception as e:
            print(f"⚠️  Erreur RSS News pour {topic} : {e}")

        return articles

    def collect_and_process(self, limit_per_topic: int = 8) -> int:
        """Collecte, enrichit et diffuse les actualités."""
        print(f"\n📰 Démarrage de la collecte d'ACTUALITÉS en direct...")
        total = 0
        for topic in ["tech", "finance"]:
            raw_articles = self.fetch_rss_news(topic, limit=limit_per_topic)
            print(f"   📥 Section {topic:<10} : {len(raw_articles)} articles récupérés")

            for article in raw_articles:
                cleaned = clean_text(article["text"])
                article["text_clean"] = cleaned

                # Inférence RoBERTa
                res = analyzer.analyze_one(cleaned)
                article["predicted_sentiment"] = res["predicted_sentiment"]
                article["sentiment_score"] = res["sentiment_score"]
                article["polarity"] = res["polarity"]
                article["topic_id"] = -1
                article["topic_label"] = topic.capitalize()

                # Stream & DB
                broker.publish(article)
                save_post(article)
                total += 1

            time.sleep(1)

        print(f"✅ Actualités traitées : {total} articles streamés et enregistrés !")
        return total


news_collector = NewsCollector()

if __name__ == "__main__":
    news_collector.collect_and_process(limit_per_topic=5)
