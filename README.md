# Titanic ML Service

Ce projet reprend le célèbre challenge Kaggle Titanic dans une démarche de data science et de ML engineering.

L'objectif est double :

* obtenir les meilleures performances possibles sur le problème de prédiction de survie ;
* construire un pipeline de machine learning propre, reproductible et industrialisable.

Le projet couvre actuellement :

* data cleaning ;
* feature engineering ;
* analyse exploratoire des données (EDA) ;
* sélection de variables ;
* benchmark et comparaison de modèles ;
* entraînement et évaluation ;
* génération de soumissions Kaggle ;
* automatisation via Makefile et UV.

Les étapes de tuning avancé, d'explicabilité et d'exposition via API seront réalisées dans un second temps.

---

## Structure du repository

```text
titanic-ml-service/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── responses/
│   └── submissions/
│
├── notebooks/
│   ├── 1.0-tc-data-collection.ipynb
│   ├── 2.0-tc-data-cleaning.ipynb
│   ├── 3.0-tc-feature-engineering.ipynb
│   ├── 4.0-tc-eda.ipynb
│   └── 5.0-tc-feature-selection.ipynb
│
├── scripts/
│   ├── build_dataset.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── models/
│   └── model.joblib
│
├── src/
│   └── titanic/
│       ├── cleaning.py
│       ├── data.py
│       ├── eda.py
│       ├── evaluate.py
│       ├── features.py
│       ├── imputations_age.py
│       ├── models.py
│       ├── params.py
│       ├── predict.py
│       ├── preprocessing.py
│       ├── selection.py
│       ├── tuning.py
│       └── validation.py
│
└── reports/
    ├── figures/
    └── metrics/
```

---

## Pipeline actuel

Le pipeline est organisé selon les étapes suivantes :

```text
Raw Data → Cleaning → Feature Engineering → Feature Selection → Preprocessing → Training → Prediction → Evaluation → Model Registry → FastAPI
```

---

## Construction du dataset

Le dataset est généré à partir des fichiers Kaggle originaux (`train.csv` et `test.csv`).

```bash
make build
```

Cette commande :

* charge les données brutes ;
* fusionne train et test ;
* applique les étapes de nettoyage ;
* construit les variables dérivées ;
* réalise les imputations nécessaires ;
* sauvegarde le dataset final dans :

```text
data/processed/titanic_features.parquet
```

---

## Entraînement

```bash
make train
```

Cette commande :

* charge les features ;
* applique la sélection de variables ;
* construit le pipeline de preprocessing ;
* benchmark plusieurs modèles ;
* sélectionne automatiquement le meilleur ;
* sauvegarde le modèle entraîné.

Les modèles actuellement comparés sont :

* Logistic Regression
* SVC
* KNN
* Decision Tree
* Random Forest
* Extra Trees
* AdaBoost
* Bagging
* Gradient Boosting
* XGBoost
* Gaussian Naive Bayes

---

## Génération des prédictions

```bash
make predict
```

Cette commande génère :

```text
data/submissions/submission.csv
```

au format attendu par Kaggle.

---

## Évaluation locale

Le projet contient les réponses du jeu de test afin d'évaluer localement les performances sans soumettre systématiquement sur Kaggle.

```bash
make evaluate
```

Exemple :

```text
Local Kaggle Score: 0.7727
```

---

## Workflow complet

```bash
make all
```

équivaut à :

```bash
make build
make train
make predict
make evaluate
```

---

## Notebooks

Les notebooks servent uniquement à l'exploration et à l'analyse.

Ils documentent notamment :

* le nettoyage des données ;
* les choix d'imputation ;
* la création des variables ;
* l'analyse exploratoire ;
* la sélection de variables.

La logique métier et le pipeline de production sont implémentés dans les modules Python du dossier `src/titanic`.

---

## Pistes d'amélioration

* hyperparameter tuning ;
* sélection automatique de variables ;
* interprétabilité (SHAP) ;
* suivi des expérimentations ;
* API FastAPI ;
* conteneurisation Docker ;
* déploiement cloud.
