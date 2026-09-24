"""
cleaner.py — Pipeline de nettoyage et normalisation NLP
Real-Time Sentiment Intelligence Platform
"""

import re
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def clean_text(text: str) -> str:
    """
    Nettoie et normalise un texte pour l'analyse de sentiment et le topic modeling.
    
    1. Passage en minuscules
    2. Suppression des URLs
    3. Suppression des mentions (@user, u/user)
    4. Suppression des références aux subreddits (r/xxx)
    5. Suppression des emojis et symboles spéciaux
    6. Conservation des ponctuations porteuses de sentiment (! ? ')
    7. Réduction des espaces multiples
    """
    if not text or not isinstance(text, str):
        return ""

    # Minuscules
    text = text.lower()

    # URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Mentions Reddit (u/xxx) et Twitter (@xxx)
    text = re.sub(r"(?:@|u/)\w+", "", text)

    # Subreddits (r/xxx)
    text = re.sub(r"r/\w+", "", text)

    # Emojis & symboles pictographiques
    text = re.sub(
        r"[\U0001F600-\U0001F64F"   # emoticons
        r"\U0001F300-\U0001F5FF"     # symbols & pictographs
        r"\U0001F680-\U0001F6FF"     # transport & map
        r"\U0001F1E0-\U0001F1FF"     # flags
        r"\U00002702-\U000027B0"     # dingbats
        r"\U0001FA00-\U0001FA6F"     # chess symbols
        r"\U0001FA70-\U0001FAFF"     # symbols extended
        r"\U00002600-\U000026FF"     # misc symbols
        r"]+", "", text
    )

    # Conservation lettres, chiffres, espaces, et ponctuation de sentiment
    text = re.sub(r"[^a-zA-Z0-9\s!?']", " ", text)

    # Espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    test_raw = "Check this out https://example.com @elonmusk r/wallstreetbets Doge to the moon! 🚀🚀 Best day ever!!!"
    print("Test brut    :", test_raw)
    print("Test nettoyé :", clean_text(test_raw))
