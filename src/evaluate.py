# src/evaluate.py 
import pandas as pd 
import joblib 
from sklearn.metrics import classification_report 
import os 
 
if __name__ == "__main__": 
    # Charger les données de test 
    test_df = pd.read_csv(os.path.join('data', 'test.csv')) 
     
    X_test = test_df['text'].astype(str) 
    y_test = test_df['sentiment'] 
 
    # Évaluer le modèle de Régression Logistique 
    print("Évaluation du modèle de Régression Logistique...") 
    lr_pipeline = joblib.load(os.path.join('models', 'logistic_regression_pipeline.joblib')) 
    lr_predictions = lr_pipeline.predict(X_test) 
     
    print("\n--- Rapport de Classification (Régression Logistique) ---") 
    lr_report = classification_report(y_test, lr_predictions, target_names=['Négatif', 'Positif'], output_dict=True) 
    print(classification_report(y_test, lr_predictions, target_names=['Négatif', 'Positif'])) 
 
    # Évaluer le modèle Naive Bayes 
    print("\nÉvaluation du modèle Naive Bayes...") 
    nb_pipeline = joblib.load(os.path.join('models', 'naive_bayes_pipeline.joblib')) 
    nb_predictions = nb_pipeline.predict(X_test) 
     
    print("\n--- Rapport de Classification (Naive Bayes) ---") 
    nb_report = classification_report(y_test, nb_predictions, target_names=['Négatif', 'Positif'], output_dict=True) 
    print(classification_report(y_test, nb_predictions, target_names=['Négatif', 'Positif'])) 

    # Affichage d'un tableau comparatif 
    print("\n--- Tableau Comparatif des Performances ---") 
    results = {
        "Modèle": ["Logistic Regression", "Naive Bayes"],
        "Accuracy": [lr_report['accuracy'], nb_report['accuracy']],
        "F1-Score (Pondéré)": [
            lr_report['weighted avg']['f1-score'],
            nb_report['weighted avg']['f1-score']
        ]
    }
    results_df = pd.DataFrame(results) 
    print(results_df.to_string(index=False))