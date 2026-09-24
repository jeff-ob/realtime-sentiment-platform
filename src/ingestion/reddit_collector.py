"""
reddit_collector.py — Collecteur de données réelles Reddit (PRAW + Fallback JSON public)
Real-Time Sentiment Intelligence Platform
"""

import sys
import time
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

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

# Subreddits cibles par domaine
DEFAULT_SUBREDDITS = {
    "tech": ["technology", "programming", "artificial", "ChatGPT"],
    "finance": ["stocks", "investing", "cryptocurrency"],
    "gaming": ["gaming", "pcgaming", "Games"]
}


class RedditCollector:
    """Collecteur de posts Reddit réels avec enrichissement NLP et streaming."""

    def __init__(self):
        self.client_id = config.REDDIT_CLIENT_ID
        self.client_secret = config.REDDIT_CLIENT_SECRET
        self.user_agent = config.REDDIT_USER_AGENT or "sentiment-intelligence/1.0"
        self.praw_client = None
        self._init_praw()

    def _init_praw(self):
        """Initialise PRAW si les identifiants sont fournis."""
        if (
            self.client_id 
            and self.client_secret 
            and self.client_id != "your_client_id_here"
            and self.client_secret != "your_client_secret_here"
        ):
            try:
                import praw
                self.praw_client = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                print("✅ Connecté à l'API Reddit officielle via PRAW.")
            except Exception as e:
                print(f"⚠️  Échec initialisation PRAW ({e}) → Utilisation du mode public JSON.")
                self.praw_client = None
        else:
            print("ℹ️  Aucune clé Reddit API configurée → Mode d'ingestion public Reddit actif.")

    def fetch_public_reddit(self, subreddit: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Récupère les derniers posts publics réels d'un subreddit via le flux RSS/Atom officiel."""
        import xml.etree.ElementTree as ET
        import urllib.request

        url = f"https://www.reddit.com/r/{subreddit}/new.rss?limit={limit}"
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        posts = []
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
                tree = ET.fromstring(xml_data)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = tree.findall("atom:entry", ns)

                for entry in entries[:limit]:
                    title_elem = entry.find("atom:title", ns)
                    title = title_elem.text if title_elem is not None and title_elem.text else ""
                    id_elem = entry.find("atom:id", ns)
                    post_id = id_elem.text if id_elem is not None and id_elem.text else str(time.time())
                    author_elem = entry.find("atom:author/atom:name", ns)
                    author = author_elem.text if author_elem is not None and author_elem.text else "reddit_user"

                    if len(title) < 10:
                        continue

                    # Déterminer la catégorie
                    topic = "tech"
                    for t, subs in DEFAULT_SUBREDDITS.items():
                        if subreddit.lower() in [s.lower() for s in subs]:
                            topic = t
                            break

                    posts.append({
                        "post_id": f"reddit_{post_id.split('/')[-1]}",
                        "source": "reddit",
                        "channel": f"r/{subreddit}",
                        "author": author,
                        "text": title,
                        "topic": topic,
                        "score": 10,
                        "num_comments": 2,
                        "created_utc": datetime.now(timezone.utc).isoformat()
                    })
        except Exception as e:
            print(f"⚠️  Erreur flux RSS Reddit pour r/{subreddit} : {e}")

        return posts

    def fetch_praw_reddit(self, subreddit_str: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Récupère les posts via PRAW officiel."""
        posts = []
        try:
            sub = self.praw_client.subreddit(subreddit_str)
            for d in sub.new(limit=limit):
                title = d.title or ""
                selftext = d.selftext or ""
                full_text = f"{title}. {selftext}".strip()

                if len(full_text) < 15:
                    continue

                topic = "tech"
                for t, subs in DEFAULT_SUBREDDITS.items():
                    if d.subreddit.display_name.lower() in [s.lower() for s in subs]:
                        topic = t
                        break

                created_time = datetime.fromtimestamp(d.created_utc, tz=timezone.utc)

                posts.append({
                    "post_id": f"reddit_{d.id}",
                    "source": "reddit",
                    "channel": f"r/{d.subreddit.display_name}",
                    "author": str(d.author) if d.author else "unknown",
                    "text": full_text[:500],
                    "topic": topic,
                    "score": d.score,
                    "num_comments": d.num_comments,
                    "created_utc": created_time.isoformat()
                })
        except Exception as e:
            print(f"⚠️  Erreur PRAW ({e}) → Fallback sur mode public.")
            return self.fetch_public_reddit(subreddit_str, limit=limit)

        return posts

    def collect_and_process(self, subreddits: Optional[List[str]] = None, limit_per_sub: int = 10) -> int:
        """Collecte, nettoie, analyse le sentiment, diffuse dans Redis et persiste en DB."""
        if not subreddits:
            subreddits = ["technology", "stocks", "gaming", "artificial"]

        total_collected = 0
        print(f"\n🌐 Démarrage de la collecte de VRAIES données Reddit ({len(subreddits)} subreddits)...")

        for sub in subreddits:
            if self.praw_client:
                raw_posts = self.fetch_praw_reddit(sub, limit=limit_per_sub)
            else:
                raw_posts = self.fetch_public_reddit(sub, limit=limit_per_sub)

            print(f"   📥 r/{sub:<15} : {len(raw_posts)} posts réels récupérés")

            for post in raw_posts:
                # 1. Nettoyage NLP
                cleaned = clean_text(post["text"])
                post["text_clean"] = cleaned

                # 2. Analyse de sentiment RoBERTa en temps réel
                res = analyzer.analyze_one(cleaned)
                post["predicted_sentiment"] = res["predicted_sentiment"]
                post["sentiment_score"] = res["sentiment_score"]
                post["polarity"] = res["polarity"]
                post["topic_id"] = -1
                post["topic_label"] = post["topic"].capitalize()

                # 3. Publication dans Redis Streams
                broker.publish(post)

                # 4. Sauvegarde dans SQLite
                save_post(post)
                total_collected += 1

            time.sleep(1)  # Respect du rate-limiting

        print(f"✅ Collecte Reddit terminée : {total_collected} vrais posts traités et streamés !")
        return total_collected


# Instance globale
reddit_collector = RedditCollector()


if __name__ == "__main__":
    reddit_collector.collect_and_process(subreddits=["technology", "stocks", "gaming"], limit_per_sub=5)
