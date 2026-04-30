"""
train_classifier.py - Entraînement du classifieur sur les embeddings

Le classifieur SVM est entraîné sur les embeddings générés par l'encodeur.
"""

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


def main():
    print("Chargement des embeddings...")
    X = np.load("embeddings.npy")
    y = np.load("labels.npy")

    print(f"Embeddings: {X.shape}")
    print(f"Labels: {y.shape}")
    print(f"Classes: {np.unique(y)}")

    print("\nEncodage des labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nValidation croisee...")
    clf = SVC(probability=True, kernel="rbf", C=10, gamma="scale")
    scores = cross_val_score(clf, X, y_encoded, cv=5)
    print(f"CV Accuracy: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")

    print("\nEntrainement final...")
    clf.fit(X, y_encoded)

    print("\nSauvegarde...")
    joblib.dump(clf, "classifier.pkl")
    np.save("label_encoder.npy", label_encoder.classes_)

    print("Fichiers crees:")
    print("  - classifier.pkl")
    print("  - label_encoder.npy")


if __name__ == "__main__":
    main()
