# 🚀 API MLOps d'Analyse de Sentiment (CamemBERT & FastAPI)

Pipeline de Machine Learning de bout en bout pour la classification de sentiments sur les réseaux sociaux (Positif / Neutre / Négatif), développé avec **Hugging Face Transformers** et industrialisé via **FastAPI**.

---

## 📋 Table des matières
1. [Aperçu du projet](#-aperçu-du-projet)
2. [Architecture technique](#-architecture-technique)
3. [Structure du projet](#-structure-du-projet)

---

## 🎯 Aperçu du projet
Ce projet vise à répondre à un cas d'usage réel en entreprise : analyser automatiquement le ton de publications sur les réseaux sociaux pour surveiller l'e-réputation. 
Le modèle sous-jacent est un **CamemBERT** fine-tuné spécifiquement pour le français, offrant des scores de confiance supérieurs à **91%**.

---

## 🛠️ Architecture technique
* **Modèle de base :** `camembert-base` (Hugging Face)
* **Fine-Tuning :** Hugging Face `Trainer` avec division Train/Validation/Test (80% / 10% / 10%)
* **Framework API :** FastAPI & Pydantic
* **Serveur web :** Uvicorn
* **Métriques :** Accuracy, Précision, Rappel, F1-Score

---

## 📁 Structure du projet
```text
Classification de texte/
│
├── dataset_3000.csv                  # Jeu de données d'entraînement (3000 lignes)
├── modele_sentiment_camembert_prod/  # Modèle fine-tuné et tokeniseur sauvegardés
├── train_production.py               # Script d'entraînement et d'évaluation
├── main.py                           # Code source de l'API FastAPI
├── test_prod.py                      # Script de test unitaire du modèle
└── README.md                         # Documentation du projet


📊 Jeu de données
Le modèle a été entraîné sur un dataset synthétique de 3 000 exemples parfaitement équilibrés :

Label 0 (Négatif) : 1 000 exemples

Label 1 (Neutre) : 1 000 exemples

Label 2 (Positif) : 1 000 exemples

⚙️ Installation et Lancement
1. Activer l'environnement virtuel

envtext\Scripts\activate     # Sur Windows

2. Installer les dépendances

pip install transformers torch datasets scikit-learn fastapi uvicorn pydantic

3. Lancer l'API FastAPI

python -m uvicorn main:app --reload

L'API est alors accessible localement sur http://127.0.0.1:8000.

🔌 Utilisation de l'API
Documentation interactive (Swagger UI)
Ouvrez votre navigateur sur l'adresse :
👉 http://127.0.0.1:8000/docs

Exemple de requête (cURL)

curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "texte": "L'\''interface est vraiment très fluide et agréable à utiliser au quotidien !"
  }'


Exemple de réponse JSON

{
  "label": "Positif",
  "score": 0.9244
}
