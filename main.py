from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialisation de l'API FastAPI
app = FastAPI(
    title="API d'Analyse de Sentiment",
    description="API de classification de texte (Positif / Neutre / Négatif) basée sur CamemBERT.",
    version="1.0.0"
)

# Chargement du modèle de production au démarrage de l'API
model_path = "./modele_sentiment_camembert_prod"
classifier = pipeline("text-classification", model=model_path)

# Modèle Pydantic pour valider le format de la requête reçue
class SentimentRequest(BaseModel):
    texte: str

# Modèle Pydantic pour formater la réponse
class SentimentResponse(BaseModel):
    label: str
    score: float

@app.get("/")
def home():
    return {"message": "API d'analyse de sentiment opérationnelle !"}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    # Prédiction sur le texte envoyé
    resultat = classifier(request.texte)[0]
    return {
        "label": resultat["label"],
        "score": round(resultat["score"], 4)
    }