from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import re
from urllib.parse import urlparse

app = Flask(__name__)

# Load the model and scaler
model_path = r'C:\Users\rakes\.vscode\cybersecurity_project\cybersecurity_model\phishing_model.h5'
scaler_path = r'C:\Users\rakes\.vscode\cybersecurity_project\cybersecurity_model\phishing_scaler.pkl'

model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

# Feature extraction function
def extract_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path

    features = [
        len(url),                             # 1. URL length
        len(domain),                          # 2. Domain length
        len(path),                            # 3. Path length
        url.count('.'),                       # 4. Number of dots
        url.count('/'),                       # 5. Number of slashes
        1 if re.search(r'https?://', url) else 0,  # 6. HTTP/HTTPS
        1 if re.search(r'\d+', domain) else 0,     # 7. Presence of digits in domain
        1 if re.search(r'@', url) else 0,          # 8. Presence of '@' in URL
        1 if re.search(r'-', domain) else 0        # 9. Presence of hyphen in domain
    ]
    
    return np.array(features).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        url = data['url']
        
        # Extract features from URL
        features = extract_features(url)
        
        # Scale the features using the same scaler from training
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Get the final result
        result = "Phishing" if prediction[0][0] > 0.5 else "Legitimate"
        
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

