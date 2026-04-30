"""
generate_embeddings.py - Génération des embeddings pour le classifier

Utilise l'encodeur extrait pour générer les embeddings du dataset.
"""

import numpy as np
import tensorflow as tf


def main():
    print("Chargement de l'encodeur...")
    encoder = tf.keras.models.load_model("encoder.h5")
    
    print(f"Encodeur input: {encoder.input_shape}")
    print(f"Encodeur output: {encoder.output_shape}")

    print("\nChargement du dataset...")
    X = np.load("X_full.npy")
    y = np.load("y_full.npy")

    print(f"Dataset: {X.shape[0]} samples")
    print(f"Features shape: {X.shape}")

    print("\nGeneration des embeddings...")
    embeddings = encoder.predict(X, verbose=1)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embeddings stats:")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std: {embeddings.std():.6f}")
    print(f"  Min: {embeddings.min():.6f}")
    print(f"  Max: {embeddings.max():.6f}")

    print("\nSauvegarde...")
    np.save("embeddings.npy", embeddings)
    np.save("labels.npy", y)

    print("Fichiers crees:")
    print(f"  - embeddings.npy ({embeddings.shape})")
    print(f"  - labels.npy ({y.shape})")


if __name__ == "__main__":
    main()
