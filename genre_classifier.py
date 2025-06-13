import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ------------------- Configuration -------------------
REMOVE_RARE_GENRES = True
RARE_GENRE_THRESHOLD = 10
# -----------------------------------------------------

# Load training data (genre + plot)
def load_train_data(path):
    genres, plots = [], []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ::: ')
            if len(parts) >= 4:
                genres.append(parts[2].strip().lower())
                plots.append(parts[3].strip())
    return pd.DataFrame({'genre': genres, 'plot': plots})

# Load test data (plot only)
def load_test_data(path):
    with open(path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

# Paths
train_path = "train_data.txt"
test_path = "test_data.txt"
solution_path = "test_data_solution.txt"

# Load data
df = load_train_data(train_path)
test_plots = load_test_data(test_path)

# Remove rare genres
if REMOVE_RARE_GENRES:
    genre_counts = df['genre'].value_counts()
    df = df[df['genre'].isin(genre_counts[genre_counts >= RARE_GENRE_THRESHOLD].index)]

print("Training samples:", df.shape)

# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in split.split(df['plot'], df['genre']):
    X_train, X_val = df['plot'].iloc[train_idx], df['plot'].iloc[val_idx]
    y_train, y_val = df['genre'].iloc[train_idx], df['genre'].iloc[val_idx]

# Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))),
    ('lr', LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight='balanced'))
])

# Train
model.fit(X_train, y_train)

# Validate
val_preds = model.predict(X_val)
print("\nValidation Results:")
print(classification_report(y_val, val_preds, zero_division=0))

# Predict test
test_preds = model.predict(test_plots)
test_preds_lower = [pred.lower() for pred in test_preds]

# Save predictions
with open("test_predictions.txt", "w", encoding="utf-8") as f:
    for pred in test_preds_lower:
        f.write(pred + "\n")

# Optional: Evaluate against true labels
if os.path.exists(solution_path):
    true_labels = []
    with open(solution_path, "r", encoding="utf-8") as f:
        for line in f:
            genre = line.strip().lower()
            if genre:
                true_labels.append(genre)

    if len(true_labels) == len(test_preds_lower):
        print("\nTest Set Evaluation:")
        print(classification_report(true_labels, test_preds_lower, zero_division=0))
    else:
        print("Warning: Mismatch between prediction and label counts.")
