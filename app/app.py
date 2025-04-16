from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model_iris.pkl", "rb") as f:
    model_iris = pickle.load(f)

with open("model_house.pkl", "rb") as f:
    model_house = pickle.load(f)

@app.route("/")
def home():
    return "ML Model is Running"

@app.route("/predict/iris", methods=["POST"])
def predict():
    data = request.get_json()

    if "features" not in data:
        return jsonify({"error": "Missing 'features' key in request"}), 400

    features = data["features"]

    for i, feature_set in enumerate(features):
        if len(feature_set) != 4:
            return jsonify({"error": f"Each feature set must contain exactly 4 values. Error at index {i}."}), 400

    input_features = np.array(data["features"])
    prediction = model_iris.predict(input_features)
    probability = model_iris.predict_proba(input_features)
    return jsonify({"prediction": prediction.tolist(),
                    "confidence": probability.tolist()})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict/housing", methods=["POST"])
def predict_housing():
    data = request.get_json()

    # Validate input
    if "features" not in data:
        return jsonify({"error": "Missing 'features' key in request"}), 400

    features = data["features"]

    # List of expected columns (order matters)
    expected_columns = [
        "price", "area", "bedrooms", "bathrooms", "stories", "mainroad", 
        "guestroom", "basement", "hotwaterheating", "airconditioning", 
        "parking", "prefarea", "furnishingstatus"
    ]

    # These are the categorical columns we need to convert to numerical
    categorical_mappings = {
        "yes": 1, "no": 0,
        "furnished": 2, "semi-furnished": 1, "unfurnished": 0
    }

    inputs = []
    for i, feature_set in enumerate(features):
        if len(feature_set) != len(expected_columns):
            return jsonify({
                "error": f"Feature set at index {i} must contain {len(expected_columns)} values."
            }), 400
        
        transformed = []
        for j, val in enumerate(feature_set):
            col_name = expected_columns[j]
            if isinstance(val, str) and val.lower() in categorical_mappings:
                transformed.append(categorical_mappings[val.lower()])
            else:
                try:
                    transformed.append(float(val))
                except ValueError:
                    return jsonify({
                        "error": f"Invalid value '{val}' for column '{col_name}' at index {i}."
                    }), 400

        inputs.append(transformed[1:])  # exclude price if that's the target

    input_features = np.array(inputs)
    prediction = model_house.predict(input_features)

    return jsonify({"predicted_prices": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000) #check your port number ( if it is in use, change the port number)
