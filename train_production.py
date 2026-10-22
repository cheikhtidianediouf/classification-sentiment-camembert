import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# 1. Chargement et séparation des données (Train / Val / Test)
df = pd.read_csv("dataset_3000.csv")

# Mélange et split 80% train, 10% validation, 10% test
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size : train_size + val_size]
test_df = df.iloc[train_size + val_size :]

# Conversion en objets Dataset Hugging Face
raw_datasets = {
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df),
}

# 2. Tokenisation
model_checkpoint = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_function(examples):
    return tokenizer(
        examples["texte"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


tokenized_datasets = {
    split: ds.map(tokenize_function, batched=True)
    for split, ds in raw_datasets.items()
}

# 3. Chargement du modèle avec les noms de labels explicites
id2label = {0: "Négatif", 1: "Neutre", 2: "Positif"}
label2id = {"Négatif": 0, "Neutre": 1, "Positif": 2}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=3, id2label=id2label, label2id=label2id
)


# 4. Fonction de calcul des métriques de performance
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# 5. Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir="./resultats_prod",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# 6. Initialisation du Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("--- Début de l'entraînement ---")
trainer.train()

# 7. Évaluation sur le jeu de test final
print("\n--- Évaluation finale sur le jeu de test ---")
test_results = trainer.evaluate(tokenized_datasets["test"])
print(test_results)

# 8. Sauvegarde finale du modèle prêt pour la prod
model.save_pretrained("./modele_sentiment_camembert_prod")
tokenizer.save_pretrained("./modele_sentiment_camembert_prod")
print(
    "\nModèle sauvegardé avec succès dans './modele_sentiment_camembert_prod' !"
)