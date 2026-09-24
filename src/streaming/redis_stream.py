"""
redis_stream.py — Message Broker et streaming temps réel avec Redis Streams
Real-Time Sentiment Intelligence Platform
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from queue import Queue

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import redis

# File mémoire de secours (fallback si Redis est inaccessible)
_local_fallback_queue = Queue()


class StreamBroker:
    """Gestionnaire de flux temps réel via Redis Streams (avec fallback local)."""

    def __init__(self):
        self.stream_name = config.REDIS_STREAM_NAME
        self.client: Optional[redis.Redis] = None
        self._connected = False
        self._connect()

    def _connect(self):
        """Tente la connexion à Redis Cloud ou local."""
        try:
            if config.REDIS_URL:
                self.client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
            else:
                self.client = redis.Redis(
                    host=config.REDIS_HOST,
                    port=config.REDIS_PORT,
                    password=config.REDIS_PASSWORD or None,
                    ssl=config.REDIS_SSL,
                    decode_responses=True
                )
            self.client.ping()
            self._connected = True
            print(f"✅ Connecté à Redis Streams ({config.REDIS_STREAM_NAME})")
        except Exception as e:
            self._connected = False
            self.client = None
            print(f"⚠️  Redis inaccessible ({e}) → Bascule sur la file mémoire locale (fallback).")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def publish(self, post_data: Dict[str, Any]) -> str:
        """Publie un post dans le stream Redis (ou dans la file de secours)."""
        payload = {"data": json.dumps(post_data, default=str)}
        if self._connected and self.client:
            try:
                # Ajout dans le Redis Stream (XADD)
                msg_id = self.client.xadd(self.stream_name, payload, maxlen=10000, approximate=True)
                return str(msg_id)
            except Exception as e:
                print(f"⚠️  Échec XADD ({e}) → Sauvegarde en file locale.")
        
        # Fallback local
        _local_fallback_queue.put(post_data)
        return f"local_{time.time()}"

    def read_latest(self, count: int = 20, last_id: str = "$") -> List[Dict[str, Any]]:
        """Lit les derniers messages du flux."""
        posts = []
        if self._connected and self.client:
            try:
                # XREAD pour écouter les nouveaux messages
                entries = self.client.xread({self.stream_name: last_id}, count=count, block=1000)
                if entries:
                    for stream, messages in entries:
                        for msg_id, data in messages:
                            if "data" in data:
                                post_dict = json.loads(data["data"])
                                post_dict["_stream_id"] = msg_id
                                posts.append(post_dict)
                return posts
            except Exception as e:
                print(f"⚠️  Erreur XREAD : {e}")

        # Fallback file locale
        while not _local_fallback_queue.empty() and len(posts) < count:
            posts.append(_local_fallback_queue.get())
        return posts

    def get_stream_length(self) -> int:
        """Retourne le nombre d'éléments dans le flux."""
        if self._connected and self.client:
            try:
                return self.client.xlen(self.stream_name)
            except Exception:
                pass
        return _local_fallback_queue.qsize()


# Instance singleton
broker = StreamBroker()


if __name__ == "__main__":
    test_msg = {"id": "test_1", "text": "Testing real-time Redis streaming", "time": time.time()}
    msg_id = broker.publish(test_msg)
    print(f"Message publié avec l'ID : {msg_id}")
    print(f"Longueur du stream : {broker.get_stream_length()}")
