# Titanic ML Service

Ce projet transforme le célèbre challenge Kaggle Titanic en un service de machine learning complet.

L'objectif est de démontrer une démarche professionnelle de data science et de ML engineering, avec le meilleur score possible.

- exploration et validation des données ;
- feature engineering ;
- entraînement et comparaison de modèles ;
- explicabilité des prédictions ;
- exposition du modèle via une API FastAPI ;
- structuration du projet selon les bonnes pratiques de développement.


## Structure du repository

```
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
```


## Architecture du projet

Le pipeline est organisé selon les étapes suivantes :

Raw Data
    ↓
Validation
    ↓
Cleaning
    ↓
Feature Engineering
    ↓
Training
    ↓
Evaluation
    ↓
Model Registry
    ↓
FastAPI


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

## API

### GET /health

Vérifie que le service est disponible.

### POST /predict

Retourne la probabilité de survie d'un passager à partir de ses caractéristiques.

### POST /explain

Retourne les principaux facteurs ayant contribué à la prédiction.

### POST /what-if

Permet d'évaluer l'impact d'une modification de certaines caractéristiques du passager sur la probabilité de survie.

### GET /model-info

Retourne des informations sur le modèle actuellement déployé.