import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.pkl")
print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded:", type(model).__name__)

REQUIRED_FEATURES = ["delivery_days", "delivery_vs_estimated", "freight_value", "product_category_name", "seller_state", "payment_type_main", "seller_historical_average_rating", "is_new_seller", "num_items", "payment_value_total", "order_hour", "order_dayofweek"]
NUMERIC_FEATURES = ["delivery_days", "delivery_vs_estimated", "freight_value", "seller_historical_average_rating", "is_new_seller", "num_items", "payment_value_total", "order_hour"]
CATEGORICAL_FEATURES = ["product_category_name", "seller_state", "payment_type_main", "order_dayofweek"]

def validate_input(data):
    errors = {}
    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        errors["missing_fields"] = missing
    for field in NUMERIC_FEATURES:
        if field in data:
            try:
                float(data[field])
            except (ValueError, TypeError):
                errors[field] = "expected numeric"
    if "freight_value" in data:
        try:
            if float(data["freight_value"]) < 0:
                errors["freight_value"] = "must be positive"
        except: pass
    if "payment_value_total" in data:
        try:
            if float(data["payment_value_total"]) < 0:
                errors["payment_value_total"] = "must be positive"
        except: pass
    return len(errors) == 0, errors

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "loaded", "model_type": type(model).__name__, "n_features": len(REQUIRED_FEATURES)}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400
    is_valid, errors = validate_input(data)
    if not is_valid:
        return jsonify({"error": "Invalid input", "details": errors}), 400
    try:
        df = pd.DataFrame([data])[REQUIRED_FEATURES]
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]
        return jsonify({"prediction": int(pred), "probability": round(float(proba), 4), "label": "positive" if pred == 1 else "negative"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array"}), 400
    if len(data) > 100:
        return jsonify({"error": "Max 100 records per batch"}), 400
    results = []
    for i, record in enumerate(data):
        is_valid, errors = validate_input(record)
        if not is_valid:
            return jsonify({"error": "Invalid input at record {}".format(i), "details": errors}), 400
        df = pd.DataFrame([record])[REQUIRED_FEATURES]
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]
        results.append({"prediction": int(pred), "probability": round(float(proba), 4), "label": "positive" if pred == 1 else "negative"})
    return jsonify({"predictions": results, "count": len(results)}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
