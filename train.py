from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Fetch dataset (Breast Cancer Wisconsin Diagnostic)
dataset = fetch_ucirepo(id=17)

X = pd.DataFrame(dataset.data.features)
y = dataset.data.targets.values.ravel()


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Model Accuracy:", acc)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully as model.pkl")
