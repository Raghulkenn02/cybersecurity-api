import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# ✅ Correct path to your preprocessed data file
data_path = r"C:\Users\rakes\OneDrive\Desktop\Raghul\Clg\Dissertation\Datasets\Preprocessed data\preprocessed_phishing_data !.csv"

# ✅ Load the dataset
data = pd.read_csv(data_path)

# ✅ Print column names to verify
print("Columns in CSV:", data.columns)

# ✅ Correct feature names after verifying
columns_to_scale = [
    'UrlLength',
    'NumNumericChars',
    'HttpsInHostname',
    'NumDash',
    'NumDots',
    'AtSymbol',
    'PathLevel',
    'PathLength',
    'label'  # Check if you need to scale 'label' or exclude it
]

# ✅ Select only the required columns for scaling
X_train = data[columns_to_scale]

# ✅ Initialize and fit scaler
scaler = StandardScaler()
scaler.fit(X_train)

# ✅ Save the scaler to a .pkl file
scaler_path = r"C:\Users\rakes\.vscode\cybersecurity_project\cybersecurity_model\phishing_scaler.pkl"
with open(scaler_path, 'wb') as file:
    pickle.dump(scaler, file)

print(f"✅ Scaler saved successfully at: {scaler_path}")
