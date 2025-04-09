# SECTION 1: Introduction
# (Markdown explaining the project and purpose)
"""
# Spam Email Detection Using Machine Learning
In this notebook, we classify emails as spam or ham using the Naive Bayes algorithm.
The implementation involves data preprocessing, training, evaluation, and predictions.
"""

# SECTION 2: Import Libraries and Load Dataset
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Sample Dataset (Expand as needed)
data = [
    ("Win a free iPhone now! Click the link to claim it.", "spam"),
    ("Meeting at 4 PM. Please join the Zoom link shared earlier.", "ham"),
    ("Hurry! You've won a $500 Amazon gift card. Claim it here.", "spam"),
    ("Can we reschedule the meeting to tomorrow?", "ham"),
    ("Your loan has been approved. Click to know more.", "spam"),
    ("Lunch plans at 1 PM?", "ham")
]

# Load data into a DataFrame
df = pd.DataFrame(data, columns=["text", "label"])
print(df.head())

# SECTION 3: Data Preprocessing
# Convert text into feature vectors
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SECTION 4: Model Training
# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# SECTION 5: Evaluation
# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# SECTION 6: Predictions on New Data
# Test with new messages
new_messages = [
    "Exclusive offer just for you! Get 50% off. Click here.",
    "Don't forget our dinner plans tonight!",
    "Congratulations, you have been selected to win a free trip."
]
new_vectors = vectorizer.transform(new_messages)
new_predictions = model.predict(new_vectors)

# Display results
for msg, pred in zip(new_messages, new_predictions):
    print(f"Message: '{msg}' -> Prediction: {pred}")


