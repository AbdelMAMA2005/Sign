"""
extract_encoder.py - Extraction de l'encodeur depuis le modèle complet

L'encodeur génère les embeddings de 64 dimensions.
"""

import tensorflow as tf
from tensorflow.keras.models import Model


def main():
    print("Chargement du modele complet...")
    full_model = tf.keras.models.load_model("full_model.h5")

    print("Extraction de la couche 'embedding'...")
    
    # Vérifier que la couche existe
    layer_names = [layer.name for layer in full_model.layers]
    print(f"Couches disponibles: {layer_names}")
    
    if "embedding" not in layer_names:
        print("ERREUR: Couche 'embedding' non trouvee!")
        return
    
    encoder = Model(
        inputs=full_model.input,
        outputs=full_model.get_layer("embedding").output
    )

    # Afficher les infos
    print(f"\nEncodeur:")
    print(f"  Input shape: {encoder.input_shape}")
    print(f"  Output shape: {encoder.output_shape}")

    print("\nSauvegarde de l'encodeur...")
    encoder.save("encoder.h5")

    print("Encodeur sauvegarde: encoder.h5")


if __name__ == "__main__":
    main()
