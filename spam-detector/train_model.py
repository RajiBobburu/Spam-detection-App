import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Convert labels
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})

# Features and labels
X = df['v2']
y = df['v1']

# TF-IDF
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=3000
)

X = tfidf.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearSVC()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model + vectorizer
joblib.dump(model, 'model/spam_model.pkl')
joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')

print("✅ Model and vectorizer saved!")