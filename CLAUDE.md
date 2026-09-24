# CLAUDE.md — Contexte du projet Real-Time Sentiment Intelligence Platform

Ce fichier permet à Claude de reprendre le projet exactement là où il s'est arrêté.
À lire en entier avant de continuer.

---

## Présentation du projet

**Real-Time Sentiment Intelligence Platform** — projet data end-to-end orienté
Data Science / NLP, construit entièrement en Python avec des outils 100% gratuits.

**Objectif CV** : couvrir les profils Data Scientist, NLP Engineer et Développeur BI,
en complément du projet Smart Sales Platform (tabular data / sales analytics).
Ce projet apporte la dimension NLP + LLM + streaming qui manquait au portfolio.

**Domaines couverts** :
- NLP / Traitement du texte
- LLM / IA générative (open source, local)
- Données temps réel / streaming

**Contrainte forte** : 100% gratuit. Aucune API payante (pas d'OpenAI, pas d'Anthropic API).
Alternatives gratuites choisies : Ollama + Mistral 7B (local) et/ou Groq API (cloud gratuit).

**Déploiement en ligne** : Streamlit Cloud (https://realtime-sentiment-platform.streamlit.app/)
Repo GitHub : https://github.com/jeff-ob/realtime-sentiment-platform

---

## Contexte portfolio

Ce projet est le **deuxième projet** du portfolio. Le premier est :

**Smart Sales Analytics Platform** (terminé, déployé)
- URL : https://jeff-ob-smart-sales-platform.streamlit.app
- Stack : pandas, scikit-learn, Prophet, KMeans, Isolation Forest, Streamlit, SQLite
- Dataset : Superstore Sales (Kaggle, 9 994 lignes)
- Profils couverts : Data Engineer, Data Scientist, Développeur BI

Le présent projet doit être **complémentaire** : NLP, LLM, streaming — rien qui
se chevauche avec le projet Superstore.

---

## Stack technique (100% gratuit)

### Ingestion
| Outil | Rôle | Notes |
|---|---|---|
| `praw` / `asyncpraw` | Collecte Reddit | API Reddit free tier, 1 000 req/min |
| `newsapi-python` | Collecte actualités | 100 req/jour (dev plan gratuit) |
| `faker` + scripts | Simulateur de flux | Pour tests offline sans quota |
| `redis-py` (Redis Streams) | Message broker | Alternative légère à Kafka, local |

### NLP / ML
| Outil | Rôle | Notes |
|---|---|---|
| `transformers` (HuggingFace) | Sentiment analysis | Modèle `cardiffnlp/twitter-roberta-base-sentiment` |
| `bertopic` | Topic modeling | Clustering sémantique non supervisé |
| `sentence-transformers` | Embeddings | Modèle `all-MiniLM-L6-v2`, local |
| `ollama` + Mistral 7B | LLM résumé / alerte | 100% local, aucune limite, aucun coût |
| Groq API (`groq`) | LLM alternatif cloud | 14 400 req/jour gratuits, ultra-rapide |

### Stockage
| Outil | Rôle | Notes |
|---|---|---|
| `sqlite3` + `sqlalchemy` | Base de données principale | Évolutif vers Postgres |
| `redis-py` | Cache temps réel | Feed live pour le dashboard |

### Visualisation
| Outil | Rôle | Notes |
|---|---|---|
| `streamlit` | Dashboard live | Déploiement Streamlit Cloud |
| `plotly` | Graphiques interactifs | |
| `jupyter` + `seaborn` | Notebook EDA | Point de départ du projet |

### Infra
| Outil | Rôle | Notes |
|---|---|---|
| `git` + GitHub | Versioning + déploiement | Public, mis en valeur sur CV |
| `venv` Python 3.11 | Environnement | |

---

## Structure du projet

```
sentiment-intelligence/
│
├── data/
│   ├── raw/                        ← données brutes collectées (non versionnées)
│   └── processed/                  ← données nettoyées
│
├── notebook/
│   └── sentiment_analysis.ipynb    ← FICHIER PRINCIPAL (brouillon de découverte)
│
├── src/
│   ├── ingestion/
│   │   ├── reddit_collector.py
│   │   ├── news_collector.py
│   │   └── simulator.py            ← tests sans API
│   ├── streaming/
│   │   └── redis_stream.py
│   ├── nlp/
│   │   ├── sentiment.py
│   │   ├── topics.py
│   │   └── llm_summary.py
│   ├── storage/
│   │   └── database.py
│   └── utils.py
│
├── dashboard/
│   └── app.py
│
├── config.py
├── requirements.txt
├── .env.example                    ← clés API Reddit / NewsAPI (jamais versionnées)
├── .gitignore
├── README.md
└── CLAUDE.md                       ← ce fichier
```

---

## Plan du notebook (fichier principal)

Le notebook est le **brouillon scientifique** du projet. Chaque modèle, chaque
feature, chaque insight doit y être exploré et validé avant d'être intégré
dans le code de production (`src/`).

**Méthode de travail convenue avec l'utilisateur :**
- Claude fournit le code d'une cellule à la fois
- L'utilisateur exécute et renvoie les résultats
- On analyse ensemble, puis on passe à la cellule suivante

### Plan des modules

| Module | Titre | Statut |
|---|---|---|
| 0 | Setup, imports, vérification environnement | ✅ Terminé |
| 1 | Collecte & exploration des données brutes | ✅ Terminé |
| 2 | Nettoyage & preprocessing NLP | ✅ Terminé |
| 3 | Analyse de sentiment (HuggingFace) | ✅ Terminé (92.75% acc) |
| 4 | Topic modeling (BERTopic) | ✅ Terminé (38 topics découverts) |
| 5 | Embeddings & visualisation sémantique | ✅ Terminé (UMAP 2D + Semantic Search) |
| 6 | Détection de tendances & anomalies | ✅ Terminé (Rolling sentiment + Z-Score spike) |
| 7 | Résumé automatique avec LLM (Ollama / Groq) | ✅ Terminé (Flash Report de crise) |
| 8 | Conclusions & recommandations pipeline | ✅ Terminé (Recherche validée) |

---

## Décisions techniques importantes

### Pourquoi Redis Streams plutôt que Kafka ?
Kafka est plus robuste en production mais nécessite une JVM et une configuration
complexe. Redis Streams offre des fonctionnalités similaires (groupes de consommateurs,
persistence, replay) avec une installation beaucoup plus simple. Pour un projet
CV solo, c'est le bon compromis. La migration vers Kafka peut être mentionnée
comme évolution future dans le README.

### Pourquoi Ollama + Mistral 7B ?
- Aucun coût, aucune limite de quota
- Fonctionne offline (pas de dépendance réseau en production)
- Mistral 7B est un modèle de qualité suffisante pour la génération de résumés courts
- Groq (Llama 3 / Mixtral) est prévu comme alternative cloud si la machine locale
  est trop limitée (Groq offre 14 400 req/jour gratuitement)

### Pourquoi `cardiffnlp/twitter-roberta-base-sentiment` ?
Modèle fine-tuné sur des données Twitter/Reddit — donc parfaitement adapté au
style informel, aux abréviations et au langage des réseaux sociaux. Meilleure
performance que les modèles génériques sur ce type de données.

### Data source strategy
On démarre avec le **simulateur** (Faker) pour construire et tester tout le pipeline
sans dépendre des quotas API. Reddit et NewsAPI sont intégrés ensuite, une fois
le pipeline validé. Cela permet aussi de travailler offline.

---

## État actuel du projet

**Phase** : Projet Terminé — Pipeline de Production & Dashboard opérationnels ! 🚀

**Réalisations validées :**
- [x] Python 3.14.7 (venv activé)
- [x] Notebook de recherche scientifique (Modules 0 à 8 validés)
- [x] Base de données SQLite (`src/storage/database.py`) avec modèles `Post` et `CrisisAlert`
- [x] Broker de streaming Redis Streams Cloud (Upstash) opérationnel (`src/streaming/redis_stream.py`)
- [x] Moteur NLP singleton RoBERTa (`src/nlp/sentiment.py`) & Cleaner (`src/nlp/cleaner.py`)
- [x] Générateur de Flash Report de crise par IA avec Groq (`src/nlp/llm_summary.py`)
- [x] Ingestion temps réel de vraies données Reddit (`src/ingestion/reddit_collector.py`)
- [x] Ingestion temps réel de vraies actualités (`src/ingestion/news_collector.py`)
- [x] Simulateur de flux continu (`src/ingestion/simulator.py`)
- [x] Dashboard interactif en direct (`dashboard/app.py`) avec KPIs, Plotly, flux live et déclencheur LLM

**Environnement vérifié (2026-09-16)** :
- [x] Python 3.14.7 (venv activé)
- [x] Jupyter installé + kernel "Sentiment Intelligence" enregistré
- [x] 18/18 librairies installées
- [x] PyTorch CPU (pas de CUDA — OK pour le projet)
- [ ] Git configuré
- [ ] (Optionnel) Ollama installé pour les modules LLM

---

## Notes importantes

### Philosophie du projet
- Le notebook est le **brouillon validé** — c'est lui qu'on montre en priorité
- Le code `src/` est l'intégration de ce qui a été validé dans le notebook
- On ne code pas en production ce qu'on n'a pas exploré dans le notebook d'abord

### Ce qu'on documente dans le notebook
Chaque module du notebook doit contenir :
1. Une cellule Markdown d'introduction (objectif, méthode)
2. Les cellules de code (une étape à la fois)
3. Une cellule Markdown de conclusion (insights trouvés, décisions prises)

### Déploiement
- Code source : GitHub (repo public, mis en valeur sur CV)
- Dashboard : Streamlit Cloud (dans la continuité du projet Superstore)
- Modèles HuggingFace : chargés à la volée (trop lourds pour être versionnés)
- Fichiers `.env` : jamais versionnés (clés API Reddit, NewsAPI)

### Fichiers à ne jamais versionner
```
data/raw/
data/processed/
*.db
.env
__pycache__/
.venv/
models/          ← modèles HuggingFace téléchargés localement
```

---

## Session de démarrage — Résumé

### Ce qui a été décidé
1. Nouveau projet différent du projet Superstore (NLP + LLM + streaming)
2. Profil cible : Data Scientist
3. Domaines : NLP, LLM / IA générative, données temps réel
4. Contrainte forte : 100% gratuit (pas d'API payante)
5. Méthode : notebook cellule par cellule, résultats partagés par l'utilisateur
6. Déploiement : GitHub + hébergement (Streamlit Cloud probable)
7. LLM retenu : Ollama + Mistral 7B (local) + Groq (cloud, backup)
8. Broker retenu : Redis Streams (plus léger que Kafka pour un projet CV)

### Ce qui reste à confirmer
- Version Python installée
- Jupyter disponible
- Ollama installé ou non (conditionne le module 7)
- Choix du sujet de données Reddit (tech ? finance ? sport ? à décider en module 1)
