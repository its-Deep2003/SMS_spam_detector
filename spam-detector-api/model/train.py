import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load dataset
data = pd.read_csv("../data/spam.csv", encoding="latin-1")

# Keep only required columns
data = data[["v1", "v2"]]

# Rename columns (clean practice)
data.columns = ["label", "text"]

# Features & labels
X = data["text"]
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ML Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved successfully")