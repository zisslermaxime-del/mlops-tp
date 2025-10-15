# src/evaluate.py
import os
import sys
import pandas as pd
import joblib
from sklearn.metrics import classification_report

DATA_PATH = os.path.join("data", "test.csv")
MODEL_DIR = "models"
LR_MODEL = os.path.join(MODEL_DIR, "logistic_regression_pipeline.joblib")
NB_MODEL = os.path.join(MODEL_DIR, "naive_bayes_pipeline.joblib")

def safe_exit(msg: str):
    print(f"[evaluate] {msg}")
    # On sort avec code 0 pour ne pas faire échouer le pipeline
    sys.exit(0)

if __name__ == "__main__":
    # 1) Vérifs préalables
    if not os.path.exists(DATA_PATH):
        safe_exit(f"Fichier de test introuvable: {DATA_PATH}. Exécutez le prétraitement avant l'évaluation.")

    missing_models = [p for p in (LR_MODEL, NB_MODEL) if not os.path.exists(p)]
    if missing_models:
        safe_exit("Modèles absents: "
                  + ", ".join(missing_models)
                  + ". Lancez d'abord l'entraînement (train.py) dans le workflow.")

    # 2) Chargement des données
    test_df = pd.read_csv(DATA_PATH)
    if "text" not in test_df.columns or "sentiment" not in test_df.columns:
        safe_exit(f"Colonnes attendues manquantes dans {DATA_PATH}. Colonnes trouvées: {list(test_df.columns)}")

    X_test = test_df["text"].astype(str)
    y_test = test_df["sentiment"]

    # 3) Évaluation Régression Logistique
    print("Évaluation du modèle de Régression Logistique...")
    try:
        lr_pipeline = joblib.load(LR_MODEL)
        lr_predictions = lr_pipeline.predict(X_test)
        print("\n--- Rapport de Classification (Régression Logistique) ---")
        lr_report = classification_report(
            y_test, lr_predictions, target_names=["Négatif", "Positif"], output_dict=True
        )
        print(classification_report(y_test, lr_predictions, target_names=["Négatif", "Positif"]))
    except Exception as e:
        safe_exit(f"Impossible de charger/évaluer le modèle LR ({LR_MODEL}): {e}")

    # 4) Évaluation Naive Bayes
    print("\nÉvaluation du modèle Naive Bayes...")
    try:
        nb_pipeline = joblib.load(NB_MODEL)
        nb_predictions = nb_pipeline.predict(X_test)
        print("\n--- Rapport de Classification (Naive Bayes) ---")
        nb_report = classification_report(
            y_test, nb_predictions, target_names=["Négatif", "Positif"], output_dict=True
        )
        print(classification_report(y_test, nb_predictions, target_names=["Négatif", "Positif"]))
    except Exception as e:
        safe_exit(f"Impossible de charger/évaluer le modèle NB ({NB_MODEL}): {e}")

    # 5) Comparatif
    print("\n--- Tableau Comparatif des Performances ---")
    results = {
        "Modèle": ["Logistic Regression", "Naive Bayes"],
        "Accuracy": [lr_report["accuracy"], nb_report["accuracy"]],
        "F1-Score (Pondéré)": [
            lr_report["weighted avg"]["f1-score"],
            nb_report["weighted avg"]["f1-score"],
        ],
    }
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
