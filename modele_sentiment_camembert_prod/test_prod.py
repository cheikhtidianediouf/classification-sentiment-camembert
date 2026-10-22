from transformers import pipeline

# Charger le pipeline avec votre modèle de production
model_path = "."
classifier = pipeline("text-classification", model=model_path)

# Tester quelques phrases variées
textes = [
    "Le service client a été absolument fantastique, merci !",
    "C'est un produit correct, sans plus.",
    "Une honte, le produit est arrivé complètement cassé."
]

print("--- Résultats des tests du modèle de production ---")
for t in textes:
    resultat = classifier(t)
    print(f"Texte : \"{t}\"")
    print(f"-> Résultat : {resultat}\n")