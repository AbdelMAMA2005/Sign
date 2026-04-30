"""
train_full_model.py - Entraînement du modèle complet

Architecture: Encodeur (génère embeddings) + Classifieur
"""

import os
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pipeline_utils import build_feature_vector_normalized, INPUT_SIZE

DATA_DIR = "data"
EMBED_SIZE = 64


def extract_hand_from_filename(filename):
    """Extrait l'indicateur de main du nom du fichier."""
    basename = os.path.basename(filename)
    match = re.search(r'_(left|right|both)_\d+\.npy$', basename)
    if match:
        return match.group(1)
    return None


def find_all_npy_files(data_dir):
    """
    Trouve tous les fichiers .npy dans le dossier data,
    peu importe la profondeur de la structure.
    """
    files_with_labels = []
    
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        
        for item in os.listdir(label_path):
            item_path = os.path.join(label_path, item)
            
            if item.endswith(".npy"):
                hand_used = extract_hand_from_filename(item)
                files_with_labels.append((item_path, label, hand_used))
            elif os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    if subitem.endswith(".npy"):
                        subitem_path = os.path.join(item_path, subitem)
                        hand_used = extract_hand_from_filename(subitem)
                        files_with_labels.append((subitem_path, label, hand_used))
    
    return files_with_labels


def load_dataset():
    """
    Charge les données directement en utilisant build_feature_vector
    pour assurer la cohérence avec le pipeline de prédiction.
    """
    X = []
    y = []

    files_with_labels = find_all_npy_files(DATA_DIR)
    print(f"Fichiers trouves: {len(files_with_labels)}")

    for filepath, label, hand_used in files_with_labels:
        try:
            sample = np.load(filepath, allow_pickle=True).item()

            left_raw = sample.get("left_hand", np.zeros((21, 3)))
            right_raw = sample.get("right_hand", np.zeros((21, 3)))

            if left_raw.shape != (21, 3):
                left_raw = np.zeros((21, 3))
            if right_raw.shape != (21, 3):
                right_raw = np.zeros((21, 3))

            # Pipeline commun NORMALISÉ avec OPTION 1
            vec, _, _ = build_feature_vector_normalized(left_raw, right_raw, hand_used=hand_used)

            if vec is None:
                continue

            X.append(vec[0])
            y.append(label)

        except Exception as e:
            print(f"Erreur {filepath}: {e}")
            continue

    return np.array(X), np.array(y)


def build_full_model(num_classes):
    """
    Modèle encodeur + classifieur pour la génération d'embeddings.
    
    Architecture améliorée avec BatchNormalization pour
    une meilleure stabilité.
    """
    inp = Input(shape=(INPUT_SIZE,))

    # Encodeur
    x = Dense(256, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Couche embedding nommée pour extraction
    embedding = Dense(EMBED_SIZE, activation='linear', name="embedding")(x)

    # Classifieur
    out = Dense(num_classes, activation='softmax')(embedding)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    print("Chargement du dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("Aucune donnee trouvee!")
        return
    
    print(f"\nDataset charge: {X.shape[0]} samples, {len(np.unique(y))} classes")
    print(f"Shape: {X.shape}")
    print(f"Input size attendu: {INPUT_SIZE}")

    # Sauvegarder pour référence
    np.save("X_full.npy", X)
    np.save("y_full.npy", y)

    # Encoding des labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, shuffle=True, stratify=y_encoded, random_state=42
    )

    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]

    # Entraînement
    model = build_full_model(num_classes=y_cat.shape[1])
    model.summary()

    print("\nEntrainement du modele...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Évaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    # Sauvegarde
    print("Sauvegarde...")
    model.save("full_model.h5")
    np.save("label_encoder.npy", label_encoder.classes_)

    print("\nFichiers crees:")
    print("  - full_model.h5")
    print("  - label_encoder.npy")
    print("  - X_full.npy")
    print("  - y_full.npy")


if __name__ == "__main__":
    main()
