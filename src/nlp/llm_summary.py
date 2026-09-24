"""
llm_summary.py — Synthèses exécutives et alertes de crise avec Groq
Real-Time Sentiment Intelligence Platform
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from groq import Groq
from src.storage.database import save_crisis_alert


def generate_crisis_report(
    crisis_posts: List[Dict[str, Any]],
    alert_date: Optional[datetime] = None,
    negative_ratio: float = 0.0
) -> Optional[str]:
    """
    Génère un Flash Report exécutif de crise via Groq (Llama 3 / GPT-OSS 120B)
    et l'enregistre en base de données.
    """
    api_key = config.GROQ_API_KEY
    if not api_key or api_key == "your_groq_key_here":
        print("⚠️  GROQ_API_KEY non configurée dans .env")
        return None

    if not crisis_posts:
        return None

    date_str = (alert_date or datetime.utcnow()).strftime("%Y-%m-%d")
    complaints = "\n".join([f"- {p.get('text', p.get('text_raw', ''))}" for p in crisis_posts[:30]])

    system_prompt = (
        "Tu es le Chief Data Officer et Responsable Qualité d'une plateforme d'intelligence temps réel. "
        "Ton rôle est d'analyser les alertes remontées par les utilisateurs et de fournir des synthèses "
        "décisionnelles limpides, percutantes et actionnables pour le comité de direction."
    )

    user_prompt = f"""
Voici les réclamations et posts négatifs collectés lors du pic d'alerte critique du {date_str} (Taux de négativité : {negative_ratio:.1f}%) :

{complaints}

Génère un **FLASH REPORT EXÉCUTIF DE CRISE** rédigé en français avec la structure suivante :
1. 🚨 **DIAGNOSTIC GLOBAL** (2-3 phrases sur l'ampleur et la nature de la crise)
2. 🔍 **RACINES DU PROBLÈME (Top Sujets Incriminés)** (lister les produits/sujets concernés et pourquoi les utilisateurs sont mécontents)
3. 💬 **VERBATIMS CLIENTS CLÉS** (2 ou 3 citations percutantes extraites des posts)
4. 🛠️ **PLAN D'ACTION IMMÉDIAT (3 RECOMMANDATIONS PRIORITAIRES)** (mesures concrètes à prendre par les équipes tech/produit/com)
"""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-120b",
            temperature=0.2,
            max_tokens=800
        )
        report = response.choices[0].message.content

        # Sauvegarder l'alerte et son rapport en base de données
        save_crisis_alert(
            alert_date=alert_date or datetime.utcnow(),
            negative_ratio=negative_ratio,
            post_count=len(crisis_posts),
            llm_report=report
        )
        print(f"✅ Alerte de crise et rapport sauvegardés en base pour le {date_str} !")
        return report
    except Exception as e:
        print(f"❌ Erreur lors de l'appel Groq : {e}")
        return None


def generate_topic_deep_dive(
    subject: str,
    period_label: str,
    posts_data: List[Dict[str, Any]],
    metrics: Dict[str, Any]
) -> Optional[str]:
    """
    Génère un Dossier d'Intelligence d'Opinion & d'Analyse Qualitative approfondie via Groq
    (polarité dominante, facteurs de plébiscite, points de friction, controverses, verbatims).
    """
    api_key = config.GROQ_API_KEY
    if not api_key or api_key == "your_groq_key_here":
        print("⚠️  GROQ_API_KEY non configurée dans .env")
        return None

    if not posts_data:
        return None

    # Extraire les exemples de posts représentatifs (max 35 pour le prompt)
    sample_posts_text = "\n".join([
        f"- [{p.get('predicted_sentiment', 'neutral').upper()} | pol:{p.get('polarity', 0.0):+.2f}] {p.get('text', '')}"
        for p in posts_data[:35]
    ])

    total = metrics.get("total_posts", len(posts_data))
    avg_pol = metrics.get("avg_polarity", 0.0)
    pos_pct = metrics.get("pos_ratio", 0.0)
    neg_pct = metrics.get("neg_ratio", 0.0)
    neu_pct = metrics.get("neu_ratio", 0.0)

    system_prompt = (
        "Tu es un Directeur d'Études d'Opinion Publique et Senior NLP Data Scientist. "
        "Ton expertise est de transformer des signaux d'opinion disparates en un diagnostic décisionnel "
        "exhaustif, percutant, argumenté et directement actionnable pour une direction générale."
    )

    user_prompt = f"""
Analyse le corpus d'avis, retours et discussions collectés sur le sujet : "{subject}" pour la période : {period_label}.

📊 DONNÉES STATISTIQUES DU MODÈLE NLP (RoBERTa) :
- Total posts analysés : {total}
- Score de Polarité Nette moyenne : {avg_pol:+.2f} (sur une échelle continue de -1.0 à +1.0)
- Répartition : {pos_pct:.1f}% Positif | {neu_pct:.1f}% Neutre | {neg_pct:.1f}% Négatif

📝 ÉCHANTILLON REPRÉSENTATIF DES VERBATIMS ET SIGNAUX :
{sample_posts_text}

RÈGLES STRICTES DE RÉDACTION :
- Rédige un véritable DOSSIER D'INTELLIGENCE D'OPINION en français avec des paragraphes développés et des arguments concrets.
- INTERDICTION FORMELLE de faire de simples listes de tirets vagues ou des phrases creuses. Chaque constat doit être justifié par le contenu des posts.
- Mets en lumière ce qui suscite l'enthousiasme, ce qui provoque de la colère/déception, et les zones de tension.

STRUCTURE ATTENDUE DU RAPPORT :

1. 📊 **THERMOMÈTRE DE POLARITÉ & PERCEPTION GLOBALE**
   - Évaluation détaillée du climat d'opinion : consensus favorable, neutralité attentiste ou crise de réputation ?
   - Analyse de la polarisation et de l'intensité émotionnelle observée dans les retours.

2. 💚 **CE QUE LES GENS ADORENT (Facteurs de Plébiscite & Satisfaction)**
   - Détail explicite et argumenté des fonctionnalités, atouts, promesses tenues ou aspects plébiscités par les utilisateurs. Expliquer concrètement POURQUOI cela plaît.

3. 💔 **CE QUE LES GENS DÉTESTENT / CRITIQUENT (Points de Friction & Frustrations)**
   - Diagnostic chirurgical des irritants majeurs : défauts, instabilités/bugs, politique tarifaire, manque d'ergonomie, sentiment de déception ou promesses non tenues.

4. ⚡ **POINTS DE CLIVAGE & DÉBATS ÉMERGENTS**
   - Quels sont les sujets qui divisent l'opinion ? (les partisans vs les détracteurs, divergences d'usage ou attentes contradictoires).

5. 💬 **VERBATIMS MARQUANTS DU CORPUS**
   - Citer 2 ou 3 phrases emblématiques extraites des retours et analyser brièvement la portée du signal qu'elles envoient.

6. 🎯 **INSIGHTS & RECOMMANDATIONS STRATÉGIQUES**
   - 3 enseignements clés et mesures prioritaires que les décideurs (produit, com, marketing) doivent immédiatement intégrer.
"""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-120b",
            temperature=0.25,
            max_tokens=1200
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Erreur lors de la synthèse thématique Groq : {e}")
        return None


if __name__ == "__main__":
    dummy_posts = [
        {"text": "Cyberpunk update broke my save file. Completely unplayable."},
        {"text": "Microtransactions are ruined in the latest patch, pure greed."}
    ]
    rep = generate_crisis_report(dummy_posts, negative_ratio=42.5)
    if rep:
        print(rep)

