import os
import numpy as np
import joblib

bundle = joblib.load("sign_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]

X = []
y = []

for letter in ["A", "B"]:
    folder = os.path.join("clean_data", letter)
    for f in os.listdir(folder):
        if f.endswith(".npy"):
            data = np.load(os.path.join(folder, f))
            X.append(data.flatten())
            y.append(letter)

X = np.array(X)
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

from collections import Counter
print("Vraies étiquettes :", Counter(y))
print("Prédictions :", Counter(y_pred))
