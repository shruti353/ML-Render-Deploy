from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Breast Cancer Classification Model is Live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    data = np.array(data).reshape(1, -1)

    prediction = model.predict(data)[0]

    result = "Malignant" if prediction == 1 else "Benign"

    return jsonify({
        "prediction": int(prediction),
        "result": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
