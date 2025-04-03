from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained model
model = tf.keras.models.load_model(r"C:\Users\rakes\OneDrive\Desktop\Raghul\Personal\Sana\Dissertation\phishing_model.h5")

# Load the scaler used in training
scaler = StandardScaler()

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    url_features = np.array(data["features"]).reshape(1, -1)  # Reshape for model
    url_features = scaler.transform(url_features)  # Standardize

    prediction = model.predict(url_features)
    result = "Phishing" if prediction[0][0] > 0.5 else "Safe"
    
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
