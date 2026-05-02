import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = "fake"
true["label"] = "real"

# Combine datasets
df = pd.concat([fake, true])

# Keep only required columns
df = df[["text", "label"]]

# Shuffle data
df = df.sample(frac=1).reset_index(drop=True)

print("📊 Dataset loaded successfully!")
print(df.head())

# Split data
x = df["text"]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# Train model
model = LogisticRegression()
model.fit(x_train, y_train)

# Evaluate model
pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)

print(f"\n✅ Model trained with accuracy: {accuracy:.2f}")

# Prediction function
def predict_news(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return prediction[0]

# User input loop
while True:
    user_input = input("\n📰 Enter news text (or type 'exit'): ")

    if user_input.lower() == "exit":
        print("👋 Exiting...")
        break

    result = predict_news(user_input)
    print(f"🔍 Prediction: {result.upper()}")