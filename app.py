from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Breast Cancer Classification Model is Live!"

# ---------- Single Prediction ----------
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

# ---------- CSV Batch Prediction ----------
@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        df = pd.read_csv(file)

        # Ensure correct shape
        if df.shape[1] != 30:
            return jsonify({
                "error": "CSV must contain exactly 30 feature columns"
            }), 400

        preds = model.predict(df)

        df["prediction"] = preds
        df["result"] = df["prediction"].map({0: "Benign", 1: "Malignant"})

        return jsonify({
            "total_samples": len(df),
            "predictions": df.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
