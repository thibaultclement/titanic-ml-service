# Projet Titanic-ML-Service

Ce dépôt contient les scripts et modèles pour la restructuration d’un cas Kaggle classique en pipeline ML complet, versionné, testable, explicable et exposé via API.
Cela concerne la problématique des survivants du Titanic. C’est un « problème jouet » mais il est traité comme un vrai mini produit ML.

## Structure du repository

titanic-ml-service/
│
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── Makefile
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── submissions/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── models/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── model_card.md
│
├── src/
│   └── titanic/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── validation.py
│       ├── features.py
│       ├── train.py
│       ├── evaluate.py
│       ├── predict.py
│       └── explain.py
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│       ├── predict.py
│       ├── explain.py
│       └── what_if.py
│
├── tests/
│   ├── test_features.py
│   ├── test_prediction.py
│   └── test_api.py
│
└── docker/
    └── Dockerfile


## Structure de la présentation (cf. portfolio web)

1. Business / ML framing
2. Data collection
3. Data validation
4. Data cleaning
5. EDA
6. Feature engineering
7. Modeling
8. Evaluation
9. Explainability
10. API / deployment
