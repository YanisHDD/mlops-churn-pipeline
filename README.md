# Customer Churn Prediction - End-to-End MLOps Pipeline

[![CI](https://github.com/YanisHDD/mlops-churn-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/YanisHDD/mlops-churn-pipeline/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E.svg)](https://scikit-learn.org/)

Projet complet MLOps pour la prediction d'attrition client (**Telco Customer Churn**).
Ce projet met en oeuvre l'ensemble des exigences du cours MLOps et les standards de production :
- Architecture modulaire sans code en dur.
- Pipeline `scikit-learn` avec `ColumnTransformer` (imputation mediane/frequente, standardisation, OneHotEncoder).
- Configuration centralisee en YAML (`configs/config.yaml`).
- Tracabilite des experiences, hyperparametres et metriques avec **MLflow** sur backend SQLite (`sqlite:///mlflow.db`).
- Gestion du cycle de vie des modeles dans le **MLflow Model Registry** avec alias `staging`.
- Generation et suivi d'artefacts d'analyse dans `reports/` (Matrice de confusion, Courbe ROC, Courbe Precision-Rappel, rapport de classification, metriques JSON).
- Inference batch (`src/predict.py`) et Microservice REST temps reel (**FastAPI** + **Pydantic** dans `service/app.py`).
- Suite de 19 tests unitaires et d'integration avec `pytest`.
- Pipeline d'integration continue (**CI/CD**) reproductible sous **GitHub Actions**.
- Conteneurisation de production avec **Docker** (`service/Dockerfile`).

---

## 1. Architecture du Projet

```text
mlops-churn-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml               # Pipeline CI GitHub Actions (qualite + pipeline)
├── configs/
│   └── config.yaml              # Configuration centralisee (donnees, features, hyperparametres)
├── data/                        # Donnees brutes (telechargees par make data, gitignored)
│   └── raw.csv
├── reports/                     # Artefacts d'evaluation (courbes, matrice, metriques, gitignored)
├── src/
│   ├── __init__.py
│   ├── download_data.py         # Telechargement reproductible du jeu Telco Churn
│   ├── pipeline.py              # Construction du Pipeline scikit-learn & ColumnTransformer
│   ├── train.py                 # Entrainement, baseline (--no-tuning), GridSearchCV et Model Registry
│   ├── evaluate.py              # Evaluation finale, generation des graphiques et metriques
│   ├── predict.py               # Inference batch en ligne de commande
│   └── utils.py                 # Utilitaires de donnees, split, MLflow et graphiques
├── service/
│   ├── __init__.py
│   ├── app.py                   # Microservice FastAPI pour l'inference temps reel
│   └── Dockerfile               # Image Docker de production pour l'API
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py         # Tests du pipeline scikit-learn et de la grille
│   ├── test_utils.py            # Tests de coercition, chargement, split et metriques
│   └── test_service.py          # Tests de l'API FastAPI (/health, /predict, /predict/batch)
├── artifacts/                   # Modeles entraines serialises (gitignored)
│   └── model.joblib
├── Dockerfile                   # Dockerfile racine pour build direct
├── Makefile                     # Automatisation (data, baseline, train, evaluate, test, serve)
├── requirements.txt             # Dependances figees avec versions exactes
├── pyproject.toml               # Configuration des outils (Ruff linter, Pytest)
├── .gitignore                   # Fichiers volumineux et caches ignores
└── README.md                    # Documentation du projet
```

---

## 2. Demarrage Rapide

### Prerequis
- Python 3.12 (ou 3.11+)
- Git

### Installation de l'environnement
```bash
# Cloner le depot
git clone https://github.com/YanisHDD/mlops-churn-pipeline.git
cd mlops-churn-pipeline

# Creer un environnement virtuel et installer les dependances
make init
# ou manuellement :
python -m venv .venv
# Windows : .venv\Scripts\activate
# Linux/macOS : source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## 3. Execution du Pipeline MLOps

Grâce au `Makefile`, toutes les etapes sont automatisees :

| Commande | Action |
|---|---|
| `make data` | Telecharge le dataset Telco Churn officiel dans `data/raw.csv` |
| `make baseline` | Entrainement rapide de reference sans recherche d'hyperparametres |
| `make train` | Entrainement complet avec `GridSearchCV` et enregistrement dans le Model Registry |
| `make evaluate` | Evaluation sur le test set, generation des courbes et metriques dans `reports/` |
| `make predict` | Inference par lot sur `data/raw.csv` -> `reports/predictions_batch.csv` |
| `make test` | Execution des 19 tests unitaires avec `pytest` |
| `make lint` | Verification du style et des imports avec `ruff` |
| `make format` | Formatage automatique du code avec `ruff` |
| `make ui` | Demarrage du serveur MLflow UI sur `http://localhost:5000` |
| `make serve` | Demarrage de l'API FastAPI sur `http://localhost:8000` |
| `make all` | Enchaine tout le cycle : `data` -> `train` -> `evaluate` -> `test` |
| `make docker-build` | Construction de l'image Docker de l'API |
| `make docker-run` | Lancement du conteneur sur le port 8000 |

---

## 4. Suivi des Experiences & Model Registry avec MLflow

Pour visualiser les runs d'entrainement, les metriques et les modeles :

```bash
make ui
# ou
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Rendez-vous sur **`http://localhost:5000`** :
- Comparaison des hyperparametres testes par `GridSearchCV`.
- Model Registry : modele versionne `ChurnClassifier` avec alias `staging`.
- Artefacts du run d'evaluation (`roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `metrics.json`).

---

## 5. Microservice FastAPI (Inference Temps Reel)

### Lancer l'API
```bash
make serve
# ou
uvicorn service.app:app --host 0.0.0.0 --port 8000 --reload
```

Documentation interactive Swagger : **`http://localhost:8000/docs`**.

### Tester une prediction (cURL)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Female",
       "SeniorCitizen": 0,
       "Partner": "Yes",
       "Dependents": "No",
       "tenure": 1,
       "PhoneService": "No",
       "MultipleLines": "No phone service",
       "InternetService": "DSL",
       "OnlineSecurity": "No",
       "OnlineBackup": "Yes",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "No",
       "StreamingMovies": "No",
       "Contract": "Month-to-month",
       "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check",
       "MonthlyCharges": 29.85,
       "TotalCharges": 29.85
     }'
```

**Reponse JSON :**
```json
{
  "churn_probability": 0.6241,
  "churn": true,
  "label": "Yes"
}
```

---

## 6. Integration Continue (CI/CD GitHub Actions)

Le workflow `.github/workflows/ci.yml` s'execute a chaque `push` et `pull_request` sur `main` :

1. **`qualite` (Lint + tests)** :
   - Execution de `ruff check` et `ruff format --check`.
   - Execution des 19 tests unitaires `pytest`.
2. **`pipeline` (Entrainement + evaluation de bout en bout)** :
   - Telechargement du jeu de donnees.
   - Entrainement complet avec `GridSearchCV` et suivi MLflow.
   - Evaluation sur le jeu de test hold-out.
   - Affichage des metriques du run.
   - Sauvegarde et televersement automatique des artefacts de rapport (`reports/*.png`, `reports/*.txt`, `reports/*.json`).

---

## 7. Conteneurisation avec Docker

```bash
# Construire l'image de l'API
make docker-build

# Executer le conteneur
make docker-run
```
L'API est joignable sur `http://localhost:8000/docs`.
