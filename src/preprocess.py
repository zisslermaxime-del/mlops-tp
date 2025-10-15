# src/preprocess.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
import os
import sys

# --- Ressources NLTK nécessaires (sans punkt/punkt_tab) ---
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "stopwords" else "corpora/stopwords")
    except LookupError:
        nltk.download(pkg)

# --- Instances globales (perf) ---
tknzr = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """
    Nettoie et prétraite un texte de tweet.
    Étapes : lower, remove URLs/hashtags/mentions, keep letters/spaces,
    tokenisation Tweet, stopwords, lemmatisation.
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    # Lower déjà géré par TweetTokenizer avec preserve_case=False,
    # mais on normalise aussi ici pour les regex :
    text = text.lower()

    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Mentions (TweetTokenizer strip_handles=True les enlève, mais on sécurise)
    text = re.sub(r"@\w+", " ", text)

    # Hashtags : on enlève juste le # mais on garde le mot
    text = text.replace("#", " ")

    # Caractères non alphabétiques
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenisation (ne dépend pas de punkt/punkt_tab)
    tokens = tknzr.tokenize(text)

    # Stop words
    tokens = [w for w in tokens if w and w not in stop_words]

    # Lemmatisation
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    # Nettoyage espaces multiples
    return " ".join(tokens).strip()

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valide/normalise les colonnes attendues :
    - Texte : 'text' (accepte 'sentence' ou 'cleaned_text' comme fallback)
    - Label : 'sentiment' (accepte 'label' ou 'target' comme fallback)
    """
    text_col_candidates = ["text", "sentence", "cleaned_text"]
    sent_col_candidates = ["sentiment", "label", "target"]

    text_col = next((c for c in text_col_candidates if c in df.columns), None)
    sent_col = next((c for c in sent_col_candidates if c in df.columns), None)

    missing = []
    if text_col is None:
        missing.append(f"texte manquant (cherché dans {text_col_candidates})")
    if sent_col is None:
        missing.append(f"sentiment/label manquant (cherché dans {sent_col_candidates})")

    if missing:
        raise ValueError(
            "Colonnes requises manquantes : " + ", ".join(missing) +
            f". Colonnes présentes : {list(df.columns)}"
        )

    # Renomme pour standardiser
    df = df.rename(columns={text_col: "text", sent_col: "sentiment"})
    return df[["text", "sentiment"]]

if __name__ == "__main__":
    raw_data_path = os.path.join("data", "raw_tweets.csv")
    if not os.path.exists(raw_data_path):
        print(f"Fichier introuvable : {raw_data_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(raw_data_path)

    # Normaliser les colonnes attendues
    df = ensure_columns(df)

    print("Début du prétraitement du texte...")
    # Sécurise NaN et applique le prétraitement
    df["text"] = df["text"].fillna("")
    df["cleaned_text"] = df["text"].apply(preprocess_text)
    print("Prétraitement terminé.")

    # Jeu final : sentiment + cleaned_text
    df_processed = df[["sentiment", "cleaned_text"]].copy()

    # Split train/test (stratifié si possible)
    X = df_processed["cleaned_text"]
    y = df_processed["sentiment"]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Si la stratification échoue (classe rare, etc.), on split sans stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Sauvegardes
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    train_df = pd.DataFrame({"text": X_train, "sentiment": y_train})
    test_df = pd.DataFrame({"text": X_test, "sentiment": y_test})
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False, encoding="utf-8")
    print("Données d'entraînement et de test sauvegardées.")
