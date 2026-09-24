"""
simulator.py — Simulateur de flux temps réel (ingestion continue)
Real-Time Sentiment Intelligence Platform
"""

import sys
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from faker import Faker
from src.nlp.cleaner import clean_text
from src.nlp.sentiment import analyzer
from src.streaming.redis_stream import broker
from src.storage.database import save_post

fake = Faker()

TOPICS = {
    "tech": {
        "channels": ["r/technology", "r/programming", "r/artificial"],
        "templates_pos": [
            "Just tried {product} and it's absolutely amazing! Best tool I've used in years.",
            "The new {product} update is incredible. Huge productivity boost!",
            "{product} just changed the game. Highly recommend it to everyone.",
            "I'm blown away by {product}. The team really nailed this release.",
        ],
        "templates_neg": [
            "{product} is completely broken after the latest update. Terrible experience.",
            "Don't waste your time on {product}. It crashed {n} times today.",
            "The {product} team clearly doesn't care about users anymore. Uninstalling.",
            "Worst update ever from {product}. They ruined everything.",
        ],
        "templates_neu": [
            "Has anyone tried {product}? Thinking about switching.",
            "The new {product} version is out. What are your thoughts?",
            "{product} announced a new feature today. Reading the docs now.",
        ],
        "items": ["ChatGPT", "VS Code", "GitHub Copilot", "Docker", "Kubernetes", "React 22", "Ollama", "Rust"]
    },
    "finance": {
        "channels": ["r/stocks", "r/investing", "r/cryptocurrency"],
        "templates_pos": [
            "{item} is up {pct}% today! Best decision I made was buying early.",
            "Strong earnings from {item}. This stock is seriously undervalued.",
            "My {item} portfolio is looking incredible this quarter. Bullish!",
        ],
        "templates_neg": [
            "{item} just crashed {pct}%. I'm losing everything. This is awful.",
            "Sold all my {item} positions. This market is looking disastrous.",
            "{item} is a scam. Pure manipulation, stay away from this garbage.",
        ],
        "templates_neu": [
            "What do you think about {item} at current price levels?",
            "{item} earnings report coming next week. Watching closely.",
            "Market analysis: {item} is trading sideways today.",
        ],
        "items": ["Tesla", "Bitcoin", "Ethereum", "NVIDIA", "Apple", "Microsoft", "Solana", "S&P 500"]
    },
    "gaming": {
        "channels": ["r/gaming", "r/pcgaming", "r/Games"],
        "templates_pos": [
            "Just finished {item} and wow, what an absolute masterpiece! 10/10.",
            "{item} is the best game I've played all year. Stunning visuals.",
            "The {item} devs actually listen to the community. Huge respect!",
        ],
        "templates_neg": [
            "{item} is a broken mess at launch. Don't buy until patched.",
            "The microtransactions in {item} are disgusting. Pure corporate greed.",
            "Frame drops and crashes every {n} minutes in {item}. Unplayable.",
        ],
        "templates_neu": [
            "Anyone know when {item} is going on sale?",
            "Playing {item} for the first time. Any tips for a beginner?",
            "{item} roadmap announced for next season.",
        ],
        "items": ["Cyberpunk 2078", "GTA VI", "Elden Ring 2", "Starfield DLC", "Baldur's Gate 4", "Zelda"]
    }
}


def generate_single_event() -> Dict[str, Any]:
    """Génère un post enrichi en direct avec NLP et prêt pour le stream."""
    topic_key = random.choice(list(TOPICS.keys()))
    t_data = TOPICS[topic_key]
    item = random.choice(t_data["items"])
    channel = random.choice(t_data["channels"])

    # Distribution réaliste
    roll = random.random()
    if roll < 0.35:
        tmpl = random.choice(t_data["templates_pos"])
    elif roll < 0.65:
        tmpl = random.choice(t_data["templates_neg"])
    else:
        tmpl = random.choice(t_data["templates_neu"])

    raw_text = tmpl.format(
        product=item,
        item=item,
        pct=random.randint(5, 75),
        n=random.randint(2, 30)
    )

    # Preprocessing NLP
    cleaned = clean_text(raw_text)

    # Inférence RoBERTa en direct
    nlp_res = analyzer.analyze_one(cleaned)

    event = {
        "post_id": f"sim_{int(time.time() * 1000)}_{random.randint(100, 999)}",
        "source": "simulator",
        "channel": channel,
        "author": fake.user_name(),
        "text": raw_text,
        "text_clean": cleaned,
        "topic": topic_key,
        "topic_id": -1,
        "topic_label": topic_key.capitalize(),
        "predicted_sentiment": nlp_res["predicted_sentiment"],
        "sentiment_score": nlp_res["sentiment_score"],
        "polarity": nlp_res["polarity"],
        "score": max(0, int(random.expovariate(1/30))),
        "num_comments": max(0, int(random.expovariate(1/8))),
        "created_utc": datetime.utcnow().isoformat()
    }

    # 1. Publier dans Redis Streams
    stream_msg_id = broker.publish(event)
    event["_stream_id"] = stream_msg_id

    # 2. Persister dans SQLite
    save_post(event)

    return event


def run_stream_feeder(count: int = 10, delay: float = 0.5):
    """Envoie une série de posts en streaming continu."""
    print(f"🚀 Démarrage de l'ingestion temps réel ({count} posts, délai: {delay}s)...")
    for i in range(1, count + 1):
        event = generate_single_event()
        icon = "🟢" if event["predicted_sentiment"] == "positive" else "🔴" if event["predicted_sentiment"] == "negative" else "⚪"
        print(f"[{i}/{count}] {icon} [{event['channel']}] {event['predicted_sentiment'].upper():<8} ({event['sentiment_score']:.2f}) : {event['text'][:65]}...")
        time.sleep(delay)
    print("✅ Ingestion streaming terminée avec succès !")


if __name__ == "__main__":
    run_stream_feeder(count=5, delay=0.2)
