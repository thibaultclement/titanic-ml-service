# Titanic ML Service

Ce projet reprend le célèbre challenge Kaggle Titanic dans une démarche de Data Science et de ML Engineering.

L'objectif est double :

* obtenir les meilleures performances possibles sur le problème de prédiction de survie ;
* construire un pipeline de machine learning propre, reproductible et industrialisable.

Le projet couvre actuellement :

* data cleaning ;
* feature engineering ;
* analyse exploratoire des données (EDA) ;
* sélection de variables ;
* benchmark et comparaison de modèles ;
* hyperparameter tuning ;
* entraînement et évaluation ;
* génération de soumissions Kaggle ;
* automatisation via Makefile et UV.

Les étapes d'explicabilité, d'API et de déploiement seront réalisées dans un second temps.

---

## Résultats

### Modèle final

* Modèle : SVC
* Kernel : RBF
* C : 2
* Gamma : scale

### Performance

* Cross-validation : ~0.80
* Score local Kaggle : ~0.80

### Variables retenues

Les variables finales incluent notamment :

* Pclass
* Title
* HasNickname
* AgeETR
* IsChild
* FarePerPerson_log1p
* HasCabin
* IsAlone
* FamilySurvivalRate
* TicketSurvivalRate

Les variables `FamilySurvivalRate` et `TicketSurvivalRate` constituent les principales améliorations apportées en fin de projet.

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
│   ├── 5.0-tc-feature-selection.ipynb
│   └── 6.0-tc-model-selection.ipynb
│
├── scripts/
│   ├── build_dataset.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── models/
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
Raw Data
    ↓
Cleaning
    ↓
Feature Engineering
    ↓
Feature Selection
    ↓
Preprocessing
    ↓
Training
    ↓
Prediction
    ↓
Evaluation
```

Le preprocessing est intégré dans une véritable pipeline scikit-learn afin d'éviter toute fuite de données lors de la validation croisée et du tuning.

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
* construit la pipeline de preprocessing ;
* entraîne le modèle final ;
* sauvegarde la pipeline entraînée.

Le modèle sauvegardé contient :

* le preprocessing ;
* le modèle ;
* la liste des variables utilisées.

Le tout est stocké dans :

```text
models/model.joblib
```

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
Local Kaggle Score: 0.8014
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
* la sélection de variables ;
* le benchmark des modèles ;
* le tuning des hyperparamètres.

La logique métier et le pipeline de production sont implémentés dans les modules Python du dossier `src/titanic`.

---

## Technologies utilisées

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Jupyter
* UV
* Ruff

---

## Commandes utiles

```bash
make build
make train
make predict
make evaluate
make all
make lint
make format
make test
```

---

## Pistes d'amélioration

* nouvelles expérimentations de features ;
* interprétabilité des prédictions (SHAP) ;
* suivi des expérimentations ;
* API FastAPI ;
* conteneurisation Docker ;
* déploiement cloud ;
* monitoring du modèle.
