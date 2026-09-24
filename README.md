# 🧠 Real-Time Sentiment Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203%20LLM-F05A28?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Redis](https://img.shields.io/badge/Redis-Streams%20Broker-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **Plateforme d'intelligence de sentiment en temps réel (NLP & LLM)** permettant l'ingestion continue de flux textuels (Reddit, News, flux synthétique), l'analyse sémantique par Transformers (`RoBERTa`), la détection de pics de crise (Z-Score) et la génération automatisée de synthèses exécutives par IA générative (`Groq` / `Llama 3`).

---

## 🎯 Aperçu du Projet

Dans un monde où les crises de réputation et les retournements d'opinion se propagent en quelques minutes, les entreprises ont besoin d'outils capables de surveiller le web social en continu.

Cette plateforme data **end-to-end** a été conçue pour couvrir l'ensemble du cycle de vie de la donnée textuelle :
1. **Ingestion & Streaming Temps Réel** : Collecte multi-sources avec tampon de messages Redis Streams.
2. **Pipeline NLP Haute Précision** : Nettoyage regex avancé et inférence par modèle Transformer fine-tuné sur les réseaux sociaux.
3. **Détection Précoce d'Anomalies** : Algorithme de détection de crise basé sur la volatilité de la polarité et le Z-Score.
4. **Génération de Flash Reports par LLM** : Synthèses d'impact et plans d'action générés à la volée par Groq (Llama 3 / Mixtral) ou Ollama (local).
5. **Dashboard Exécutif BI** : Visualisations interactives temps réel avec Plotly et Streamlit.

---

## 🏗️ Architecture du Système

```mermaid
flowchart TD
    subgraph Ingestion ["1. Multi-Source Ingestion"]
        R[Reddit API / PRAW] --> S[Ingestion Engine]
        N[NewsAPI] --> S
        F[Faker Stream Simulator] --> S
    end

    subgraph Streaming ["2. Streaming Broker"]
        S --> RS[(Redis Streams Broker)]
        RS -. Fallback mémoire .-> MQ[In-Memory Queue]
    end

    subgraph Processing ["3. NLP & AI Processing Engine"]
        RS --> CL[Text Cleaner & Normalizer]
        MQ --> CL
        CL --> TR[RoBERTa Transformer Sentiment Engine]
        TR --> ZS[Z-Score Spike & Crisis Detector]
        ZS --> LLM[Groq Llama 3 / Ollama Mistral]
    end

    subgraph Storage ["4. Storage & Persistence"]
        TR --> DB[(SQLite / SQLAlchemy)]
        ZS --> DB
        LLM --> DB
    end

    subgraph UI ["5. Executive BI Dashboard"]
        DB --> ST[Streamlit Real-Time Dashboard]
        ST --> PL[Plotly Charts & Dynamic Filtering]
        ST --> AI[On-Demand LLM Topic Deep Dive]
    end
```

---

## ✨ Fonctionnalités Clés

### 1. Ingestion Multi-Sources & Streaming Résilient
- **Reddit API (PRAW)** : Ingestion en temps réel des discussions sur des subreddits ciblés (`technology`, `stocks`, `artificial`, etc.).
- **NewsAPI** : Surveillance des titres d'actualités financières et technologiques mondiales.
- **Simulateur continu (Faker)** : Génération de flux synthétiques réalistes pour tests hors-ligne et démonstrations continues.
- **Redis Streams (Upstash Cloud)** : Architecture de messagerie découplée avec bascule automatique sur file mémoire locale (*graceful fallback*).

### 2. Modèle NLP Transformer (`RoBERTa`)
- Utilisation du modèle `cardiffnlp/twitter-roberta-base-sentiment-latest` optimisé pour le style informel et les néologismes des réseaux sociaux.
- Score de confiance Softmax et calcul de polarité continue normalisée sur $[-1.0, +1.0]$.
- Architecture Singleton avec **chargement asynchrone / lazy loading** pour un démarrage instantané de l'interface.

### 3. Recherche Thématique & Topic Modeling
- Recherche sémantique à la volée sur n'importe quel sujet (`"Nvidia AI"`, `"Bitcoin"`, `"Cybersecurity"`).
- Nettoyage et normalisation NLP rigoureux (URLs, mentions, balises, émojis et ponctuation porteuse de tonalité).

### 4. Rapports de Crise par IA Générative (LLM)
- En cas de détection d'une anomalie négative ou sur demande de l'utilisateur, déclenchement d'un **Flash Report de crise** via **Groq** (`Llama 3-70B`) ou **Ollama** (`Mistral 7B`).
- Analyse contextuelle : causes probables, impact sur l'image de marque et recommandations d'action pour les équipes de communication.

### 5. Dashboard Interactif Streamlit
- **Mode Sombre Professionnel** avec typographie moderne (Inter & calligraphie d'accent).
- **Cartes KPIs dynamiques** : Volume total, ratio Positif / Négatif, polarité moyenne, alertes actives.
- **Visualisations Plotly** : Histogrammes de sentiment, évolution temporelle de la polarité, et distribution par source.

---

## 📂 Structure du Répertoire

```text
sentiment-intelligence/
│
├── dashboard/
│   ├── app.py                      # Dashboard principal Streamlit
│   └── static/                     # Assets statiques et styles
│
├── src/
│   ├── ingestion/
│   │   ├── reddit_collector.py     # Collecteur Reddit PRAW
│   │   ├── news_collector.py       # Collecteur NewsAPI
│   │   ├── simulator.py            # Simulateur de flux continu
│   │   └── topic_search.py         # Recherche dynamique par mot-clé
│   │
│   ├── streaming/
│   │   └── redis_stream.py         # Broker Redis Streams (Upstash + Fallback)
│   │
│   ├── nlp/
│   │   ├── cleaner.py              # Pipeline de nettoyage textuel
│   │   ├── sentiment.py            # Moteur RoBERTa (chargement lazy)
│   │   └── llm_summary.py          # Synthèses de crise Groq / Ollama
│   │
│   └── storage/
│       └── database.py             # Modèles SQLAlchemy (Post, CrisisAlert)
│
├── notebook/
│   └── sentiment_analysis.ipynb    # Notebook de recherche (EDA, BERTopic, UMAP)
│
├── config.py                       # Configuration centralisée & gestion des secrets
├── requirements.txt                # Dépendances de production (Streamlit Cloud)
├── requirements-dev.txt            # Dépendances d'exploration (Notebooks, BERTopic)
├── .env.example                    # Template des variables d'environnement
├── .gitignore                      # Fichiers exclus du versioning
└── README.md                       # Documentation du projet
```

---

## 🚀 Démarrage Rapide (Local)

### 1. Prérequis
- Python 3.11 ou supérieur
- Git

### 2. Cloner le projet & créer l'environnement virtuel

```bash
# Cloner le repository
git clone https://github.com/jeff-ob/realtime-sentiment-platform.git
cd realtime-sentiment-platform

# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement
# Sur Windows (PowerShell) :
.venv\Scripts\Activate.ps1
# Sur macOS / Linux :
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Configurer les variables d'environnement

Copiez le fichier `.env.example` en `.env` :

```bash
cp .env.example .env
```

Renseignez vos clés d'API (toutes gratuites) :

```env
# Reddit API (https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=votre_client_id
REDDIT_CLIENT_SECRET=votre_client_secret
REDDIT_USER_AGENT=sentiment-intelligence/1.0

# NewsAPI (https://newsapi.org/register)
NEWSAPI_KEY=votre_cle_newsapi

# Groq Cloud API (https://console.groq.com/keys) - Gratuit & ultra-rapide
GROQ_API_KEY=votre_cle_groq

# Redis Streams (Upstash Redis Cloud gratuit : https://upstash.com)
REDIS_URL=rediss://default:votre_token@votre_instance.upstash.io:6379
```

> **Note :** Si Redis n'est pas configuré, l'application bascule automatiquement et de manière transparente sur la file mémoire locale !

### 4. Lancer le Dashboard

```bash
streamlit run dashboard/app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`.

---

## ☁️ Déploiement sur Streamlit Cloud

1. Poussez votre code sur un dépôt GitHub public.
2. Rendez-vous sur **[share.streamlit.io](https://share.streamlit.io)** et sélectionnez votre repo.
3. Définissez le point d'entrée : `dashboard/app.py`.
4. Dans **Settings > Secrets**, ajoutez vos clés d'API au format TOML :

```toml
REDDIT_CLIENT_ID = "..."
REDDIT_CLIENT_SECRET = "..."
REDDIT_USER_AGENT = "sentiment-intelligence/1.0"
NEWSAPI_KEY = "..."
GROQ_API_KEY = "..."
REDIS_URL = "..."
```

---

## 🛠️ Stack Technique

| Domaine | Technologies |
|---|---|
| **Langage & Core** | Python 3.14 / 3.11, Pandas, NumPy |
| **Ingestion & Streaming** | PRAW (Reddit), NewsAPI, Faker, Redis Streams (Upstash) |
| **NLP & Deep Learning** | HuggingFace Transformers, PyTorch, Twitter-RoBERTa, Sentence-Transformers |
| **Topic Modeling** | BERTopic, UMAP, HDBSCAN (validés dans le notebook EDA) |
| **IA Générative / LLM** | Groq API (`Llama 3-70B`), Ollama (`Mistral 7B`) |
| **Base de Données** | SQLite, SQLAlchemy ORM |
| **Visualisation** | Streamlit, Plotly Express, Plotly Graph Objects, CSS personnalisé |

---

## 👤 Auteur & Portfolio

**Projet Portfolio Data Science / NLP / BI**

- **Projet complémentaire** : [Smart Sales Analytics Platform](https://jeff-ob-smart-sales-platform.streamlit.app) *(Sales Analytics, Prophet, Machine Learning Tabulaire)*
- **Profils ciblés** : Data Scientist, NLP Engineer, Développeur BI

---

## 📄 Licence

Ce projet est sous licence **MIT**. Consultez le fichier `LICENSE` pour plus de détails.
