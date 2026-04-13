import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def load_model():
    model_path = "model/spam_model.pkl"
    tfidf_path = "model/tfidf_vectorizer.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError("spam_model.pkl not found")

    svm_model = joblib.load(model_path)

    if os.path.exists(tfidf_path):
        tfidf = joblib.load(tfidf_path)
    else:
        if os.path.exists("spam.csv"):
            df = pd.read_csv("spam.csv", encoding="latin-1")
            tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
            tfidf.fit(df["v2"])
            joblib.dump(tfidf, tfidf_path)
        else:
            raise FileNotFoundError("tfidf_vectorizer.pkl or spam.csv required")

    return svm_model, tfidf