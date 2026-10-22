# Utilisation d'une image Python officielle légère
FROM python:3.10-slim

# Définition du dossier de travail dans le conteneur
WORKDIR /app

# Installation des dépendances système de base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code de l'API et du dossier du modèle
COPY . .

# Exposition du port de l'API
EXPOSE 8000

# Commande de démarrage de l'API via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]