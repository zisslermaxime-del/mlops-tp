# src/train.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

# Définir le nom de l’expérience MLflow
mlflow.set_experiment("Analyse de Sentiments Twitter")

def train_model(model_name, pipeline):
    """Entraîne un modèle et log les informations avec MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Charger les données
        train_df = pd.read_csv(os.path.join("data", "train.csv"))
        X_train = train_df["text"].astype(str)
        y_train = train_df["sentiment"]

        # Logger les paramètres du modèle
        params = pipeline.get_params()
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("tfidf_ngram_range", params["tfidf__ngram_range"])
        mlflow.log_param("tfidf_max_features", params["tfidf__max_features"])
        if model_name == "LogisticRegression":
            mlflow.log_param("clf_max_iter", params["clf__max_iter"])

        # Entraînement
        print(f"Entraînement du modèle {model_name}...")
        pipeline.fit(X_train, y_train)
        print("Entraînement terminé.")

        # Logger le modèle comme artefact MLflow
        mlflow.sklearn.log_model(pipeline, artifact_path=f"{model_name}_pipeline")

        print(f"Modèle {model_name} et paramètres loggés avec MLflow.")


if __name__ == "__main__":
    # --- Pipeline Régression Logistique ---
    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter=200, random_state=42))
    ])
    train_model("LogisticRegression", lr_pipeline)

    # --- Pipeline Naive Bayes ---
    nb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("clf", MultinomialNB())
    ])
    train_model("NaiveBayes", nb_pipeline)
