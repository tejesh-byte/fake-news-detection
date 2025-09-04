import re
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download stopwords + wordnet (only first run)
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# -----------------------------
# Load and prepare dataset
# -----------------------------
# Replace 'data/fake_news.csv' with your actual CSV file path
df = pd.read_csv("data/fake_news.csv")  # assuming CSV with 'text' and 'label' columns
df['cleaned_text'] = df['text'].apply(clean_text)

X = df['cleaned_text']
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorizer + Logistic Regression classifier
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

if __name__ == "__main__":
    sample = "Breaking News!!! This is a FAKE news article about politics in 2025..."
    print("Original:", sample)
    print("Cleaned:", clean_text(sample))
