# Customer Churn Prediction - End-to-End MLOps Pipeline

[![MLOps CI Pipeline](https://github.com/YanisHDD/mlops-churn-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/YanisHDD/mlops-churn-pipeline/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E.svg)](https://scikit-learn.org/)

Projet de référence MLOps pour la prédiction d'attrition client (**Telco Customer Churn**).
Ce projet implémente l'ensemble des bonnes pratiques d'ingénierie logicielle et de Machine Learning en production :
- Architecture modulaire sans code en dur.
- Pipeline `scikit-learn` complet avec `ColumnTransformer` (imputation, normalisation, encodage OneHot).
- Configuration externalisée en YAML (`configs/config.yaml`).
- Traçabilité totale des expériences, hyperparamètres et métriques avec **MLflow**.
- Génération automatique d'artefacts visuels (Matrice de confusion, Courbes ROC et Précision-Rappel).
- Microservice REST temps réel avec **FastAPI** et validation des données par **Pydantic**.
- Suite de tests unitaires automatisés avec `pytest`.
- Pipeline d'intégration continue (**CI/CD**) avec **GitHub Actions**.
- Conteneurisation prête pour le déploiement avec **Docker**.

---

## 1. Architecture du Projet

```text
mlops-churn-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml               # Pipeline CI/CD GitHub Actions (linter + tests)
├── configs/
│   └── config.yaml              # Configuration centralisée (données, features, hyperparamètres)
├── data/                        # Données (exclues de git par .gitignore)
│   ├── raw.csv                  # Dataset brut Telco Customer Churn
│   └── processed/test.csv       # Jeu de test hold-out sauvegardé pour l'évaluation
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # Construction du Pipeline scikit-learn & ColumnTransformer
│   ├── train.py                 # Entraînement, GridSearchCV et tracking MLflow
│   ├── evaluate.py              # Évaluation, calcul des métriques et artefacts graphiques
│   └── utils.py                 # Utilitaires de données, plots et chargement de configuration
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py         # Tests unitaires du pipeline et de l'imputation
│   └── test_api.py              # Tests unitaires des endpoints FastAPI
├── artifacts/                   # Modèles entraînés et figures générées (gitignored)
├── app.py                       # Microservice FastAPI pour l'inférence temps réel
├── download_data.py             # Téléchargement automatique du dataset brut
├── Dockerfile                   # Image Docker de production pour l'API
├── Makefile                     # Commandes d'automatisation (train, eval, test, ui, api)
├── requirements.txt             # Dépendances Python du projet
├── .env.example                 # Variables d'environnement pour MLflow et FastAPI
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

Grâce au `Makefile`, toutes les étapes sont exécutables en une commande :

| Commande | Action |
|---|---|
| `make data` | Télécharge le dataset Telco Churn officiel dans `data/raw.csv` |
| `make train` | Lance l'entraînement avec recherche par grille et logue tout dans MLflow |
| `make evaluate` | Évalue le modèle sur le test set et génère les graphiques d'analyse |
| `make test` | Exécute la suite complète de tests unitaires avec `pytest` |
| `make pipeline` | Exécute la chaîne complète d'un seul coup (`data` -> `train` -> `evaluate` -> `test`) |
| `make run-ui` | Démarre le serveur MLflow UI sur `http://127.0.0.1:5000` |
| `make run-api` | Démarre l'API FastAPI sur `http://127.0.0.1:8000` |

---

## 4. Suivi des Expériences avec MLflow

Pour visualiser l'historique des entraînements, les métriques (ROC-AUC, Accuracy, F1) et les artefacts :

```bash
make run-ui
# ou
mlflow ui --host 127.0.0.1 --port 5000
```
Ouvrez votre navigateur sur **`http://127.0.0.1:5000`** pour accéder au tableau de bord MLflow :
- Comparaison des hyperparamètres testés par `GridSearchCV`.
- Courbe ROC (`roc_curve.png`).
- Matrice de confusion (`confusion_matrix.png`).
- Modèle versionné avec signature d'entrée/sortie.

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

## 6. Intégration Continue (CI/CD)

Le projet intègre un pipeline GitHub Actions (`.github/workflows/ci.yml`) qui s'exécute automatiquement à chaque `push` ou `pull_request` sur la branche `main` :
1. Mise en place de l'environnement Python.
2. Installation des dépendances.
3. Vérification du format et du linting avec `ruff`.
4. Exécution de la suite de tests unitaires `pytest`.

---

## 7. Déploiement avec Docker

Pour conteneuriser le microservice :
```bash
# Construire l'image Docker
docker build -t churn-classifier-service:latest .

# Lancer le conteneur
docker run -d -p 8000:8000 --name churn-api churn-classifier-service:latest
```
L'API est accessible sur `http://localhost:8000/docs`.
