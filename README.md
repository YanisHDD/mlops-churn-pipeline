# Customer Churn Prediction - End-to-End MLOps Pipeline

[![CI](https://github.com/YanisHDD/mlops-churn-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/YanisHDD/mlops-churn-pipeline/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E.svg)](https://scikit-learn.org/)

Projet complet MLOps pour la prédiction d'attrition client (**Telco Customer Churn**).
Ce projet met en oeuvre l'ensemble des exigences du cours MLOps et les standards de production :
- Architecture modulaire sans code en dur.
- Pipeline `scikit-learn` avec `ColumnTransformer` (imputation médiane/fréquente, standardisation, OneHotEncoder).
- Configuration centralisée en YAML (`configs/config.yaml`).
- Traçabilité totale des expériences, hyperparamètres et métriques avec **MLflow** sur backend SQLite (`sqlite:///mlflow.db`).
- Versioning et gestion du cycle de vie des modèles dans le **MLflow Model Registry** (`ChurnClassifier`).
- Génération et suivi d'artefacts visuels (Matrice de confusion, Courbe ROC, Courbe Precision-Rappel).
- Script de vérification automatisé de l'API MLflow (`scripts/verify_mlflow_ui.py`).
- Inférence batch (`src/predict.py`) et Microservice REST temps réel (**FastAPI** + **Pydantic**).
- Suite de tests unitaires et d'intégration avec `pytest`.
- Pipeline d'intégration continue (**CI/CD** à 3 étapes) sous **GitHub Actions**.
- Conteneurisation avec **Docker** et publication d'image.

---

## 1. Architecture du Projet

```text
mlops-churn-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml               # Pipeline CI/CD GitHub Actions (3 étapes chaînées)
├── configs/
│   └── config.yaml              # Configuration centralisée (données, features, hyperparamètres)
├── data/                        # Données (exclues de git par .gitignore)
│   └── raw.csv                  # Dataset brut Telco Customer Churn
├── scripts/
│   └── verify_mlflow_ui.py      # Vérification headless de l'API REST MLflow UI
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # Construction du Pipeline scikit-learn & ColumnTransformer
│   ├── baseline.py              # Garde-fou rapide sans hyperparamètres
│   ├── train.py                 # Entraînement, GridSearchCV et enregistrement Registry MLflow
│   ├── evaluate.py              # Évaluation depuis le Model Registry et génération d'artefacts
│   ├── predict.py               # Inférence batch depuis le Model Registry
│   └── utils.py                 # Utilitaires de données, plots et chargement de configuration
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py         # Tests unitaires du pipeline et de l'imputation
│   ├── test_utils.py            # Tests de chargement, typage et split
│   ├── test_mlflow_tracking.py  # Tests d'intégration du tracking et du Model Registry
│   └── test_api.py              # Tests unitaires des endpoints FastAPI
├── artifacts/                   # Modèles entraînés et figures générées (gitignored)
├── app.py                       # Microservice FastAPI pour l'inférence temps réel
├── download_data.py             # Téléchargement automatique du dataset brut
├── Dockerfile                   # Image Docker de production pour l'entraînement / inférence
├── Makefile                     # Commandes d'automatisation (train, eval, test, ui, api, verify)
├── requirements.txt             # Dépendances Python du projet
├── pyproject.toml               # Configuration des outils (Ruff linter)
├── .gitignore                   # Fichiers volumineux, temporaires et secrets ignorés
└── README.md                    # Documentation du projet
```

---

## 2. Démarrage Rapide

### Prérequis
- Python 3.10+ (ou Python 3.11/3.12)
- Git

### Installation de l'environnement
```bash
# Cloner le dépôt
git clone https://github.com/YanisHDD/mlops-churn-pipeline.git
cd mlops-churn-pipeline

# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement
# Sur Windows :
.venv\Scripts\activate
# Sur Linux / macOS :
source .venv/bin/activate

# Installer les dépendances
pip install -U pip
pip install -r requirements.txt
```

---

## 3. Exécution du Pipeline MLOps

Grâce au `Makefile`, toutes les étapes sont automatisées :

| Commande | Action |
|---|---|
| `make data` | Télécharge le dataset Telco Churn officiel dans `data/raw.csv` |
| `make baseline` | Exécute un premier entraînement de référence rapide (sans tuning) |
| `make train` | Lance le GridSearchCV et enregistre le modèle dans le MLflow Registry |
| `make evaluate` | Charge le modèle depuis le Registry, calcule le ROC-AUC et génère les figures |
| `make verify-mlflow` | Démarre un serveur MLflow headless et valide son API REST |
| `make test` | Exécute la suite complète de 13 tests unitaires et d'intégration (`pytest`) |
| `make lint` | Vérifie la conformité du code avec `ruff` |
| `make pipeline` | Exécute la chaîne complète locale (`data` -> `baseline` -> `train` -> `evaluate` -> `test` -> `verify-mlflow`) |
| `make mlflow-ui` | Démarre le serveur MLflow UI sur `sqlite:///mlflow.db` |
| `make run-api` | Démarre l'API FastAPI sur `http://127.0.0.1:8000` |

---

## 4. Suivi des Expériences & Model Registry avec MLflow

Pour visualiser l'historique des entraînements, les métriques (ROC-AUC, Accuracy, F1) et les artefacts :

```bash
make mlflow-ui
# ou
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Ouvrez votre navigateur sur **`http://127.0.0.1:5000`** pour accéder au tableau de bord MLflow :
- Runs de tuning `GridSearchCV` et métriques de validation croisée.
- Versioning automatique dans le **Model Registry** sous le nom `ChurnClassifier`.
- Courbe ROC (`roc_curve.png`).
- Courbe Precision-Recall (`pr_curve.png`).
- Matrice de confusion (`confusion_matrix.png`).
- Table des prédictions d'erreur (`predictions.csv`).

---

## 5. Microservice FastAPI (Inférence Temps Réel)

### Lancer l'API
```bash
make run-api
# ou
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Accédez à la documentation Swagger interactive sur **`http://127.0.0.1:8000/docs`**.

### Tester une prédiction (cURL)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Female",
       "SeniorCitizen": "0",
       "Partner": "Yes",
       "Dependents": "No",
       "tenure": 1.0,
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

**Réponse JSON :**
```json
{
  "churn_prediction": "Yes",
  "churn_code": 1,
  "churn_probability": 0.6241,
  "risk_level": "High"
}
```

---

## 6. Intégration Continue (CI/CD GitHub Actions)

Le workflow `.github/workflows/ci.yml` est structuré en **3 étapes séquentielles** :

1. **`lint-and-test`** :
   - Vérification de la syntaxe et des normes avec `ruff`.
   - Exécution des 13 tests unitaires et d'intégration (`pytest`).
2. **`train-and-evaluate`** (dépendant de `lint-and-test`) :
   - Téléchargement du dataset officiel.
   - Exécution de la baseline.
   - Entraînement avec `GridSearchCV` et tracking SQLite MLflow.
   - Évaluation et logging des artefacts visuels.
   - Vérification de l'API REST du serveur MLflow via `verify_mlflow_ui.py`.
   - Téléversement des résultats MLflow (`mlflow.db` et `mlruns/`) sous forme d'artefact de build (`mlflow-results`).
3. **`docker-publish`** (dépendant de `train-and-evaluate`) :
   - Build de l'image Docker de production.
   - Publication automatique sur Docker Hub (si les secrets `DOCKERHUB_USERNAME` et `DOCKERHUB_TOKEN` sont configurés).

---

## 7. Déploiement avec Docker

Pour conteneuriser le service :
```bash
# Construire l'image Docker
docker build -t churn-classifier:latest .

# Lancer l'entraînement ou l'inférence
docker run --rm churn-classifier:latest
```

