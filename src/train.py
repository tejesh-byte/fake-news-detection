import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from data_prep import clean_text

def train_and_save():
    print("🚀 Training started...")

    # Load dataset
    df = pd.read_csv("data/fake_news.csv")

    # Clean text
    df["cleaned"] = df["text"].apply(clean_text)

    X = df["cleaned"]
    y = df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Save model
    joblib.dump(pipeline, "model/fake_news_model.pkl")
    print("✅ Model saved at model/fake_news_model.pkl")

if __name__ == "__main__":
    train_and_save()
