import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from model.gemini_check import verify_news

# =========================
# LOAD DATA
# =========================
fake = pd.read_csv("Fake.csv", engine="python", on_bad_lines="skip")
true = pd.read_csv("True.csv", engine="python", on_bad_lines="skip")

# =========================
# ADD LABELS
# =========================
fake["label"] = 0
true["label"] = 1

# =========================
# MERGE DATA
# =========================
data = pd.concat([fake, true])

# =========================
# KEEP REQUIRED COLUMNS
# =========================
data = data[["text", "label"]]

# REMOVE NULL VALUES
data = data.dropna()

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    return text

# APPLY CLEANING
data["text"] = data["text"].apply(clean_text)

# =========================
# TF-IDF VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(data["text"])
y = data["label"]

print("Dataset Shape:", X.shape)

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# =========================
# MODEL ACCURACY
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# =========================
# USER INPUT
# =========================
user_news = input("\nEnter news to check:\n")

# =========================
# CLEAN INPUT
# =========================
clean_input = clean_text(user_news)

# =========================
# ML PREDICTION
# =========================
news_vector = vectorizer.transform([clean_input])

prediction = model.predict(news_vector)

probability = model.predict_proba(news_vector)

confidence = max(probability[0]) * 100

# =========================
# ML RESULT
# =========================
if prediction[0] == 0:
    ml_result = "FAKE"
else:
    ml_result = "REAL"

print("\n=========================")
print("ML MODEL RESULT")
print("=========================")

print(f"Prediction : {ml_result}")
print(f"Confidence : {confidence:.2f}%")

# =========================
# GEMINI AI FACT CHECK
# =========================
print("\n=========================")
print("GEMINI AI FACT CHECK")
print("=========================")

try:

    ai_result = verify_news(user_news)

    # Gemini quota or API failure check
    if (
        "quota" in ai_result.lower()
        or "api_key_invalid" in ai_result.lower()
        or "error" in ai_result.lower()
    ):

        print("AI Verification Unavailable")
        print("Using only local ML model prediction.")

    else:

        print(ai_result)

except Exception as e:

    print("AI Verification Failed")
    print("Error:", str(e))

# =========================
# FINAL PROFESSIONAL OUTPUT
# =========================
print("\n=========================")
print("FINAL RESULT")
print("=========================")

# If AI unavailable → don't trust ML fully
try:

    if (
        "quota" in ai_result.lower()
        or "api_key_invalid" in ai_result.lower()
        or "error" in ai_result.lower()
    ):

        print("Status : NEEDS HUMAN VERIFICATION")
        print("Reason : AI fact-check unavailable")

    else:

        print("AI fact-check completed successfully.")

except:

    print("Verification process incomplete.")