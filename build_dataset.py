"""
build_dataset.py - Construction du dataset

CORRECTION: Parcours récursif des dossiers pour gérer
les deux structures possibles:
- data/{label}/*.npy (nouvelle structure)
- data/{label}/{hand}/*.npy (ancienne structure)
"""

import os
import re
import numpy as np
from pipeline_utils import build_feature_vector_normalized

DATA_DIR = "data"


def extract_hand_from_filename(filename):
    """
    Extrait l'indicateur de main du nom du fichier.
    
    Formats supportés:
    - LABEL_left_0000.npy -> "left"
    - LABEL_right_0000.npy -> "right"
    - LABEL_both_0000.npy -> "both"
    - LABEL_0000.npy -> None (auto-détection)
    
    Args:
        filename: str - nom du fichier
        
    Returns:
        str ou None - "left", "right", "both", ou None
    """
    basename = os.path.basename(filename)
    
    # Pattern: LABEL_(left|right|both)_XXXX.npy
    match = re.search(r'_(left|right|both)_\d+\.npy$', basename)
    if match:
        return match.group(1)
    
    return None


def find_all_npy_files(data_dir):
    """
    Trouve tous les fichiers .npy dans le dossier data,
    peu importe la profondeur de la structure.
    
    Returns:
        list of tuples: [(filepath, label, hand_used), ...]
        - hand_used: "left", "right", "both", ou None
    """
    files_with_labels = []
    
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        
        # Chercher les .npy directement dans data/{label}/
        for item in os.listdir(label_path):
            item_path = os.path.join(label_path, item)
            
            if item.endswith(".npy"):
                hand_used = extract_hand_from_filename(item)
                files_with_labels.append((item_path, label, hand_used))
            
            # Chercher aussi dans les sous-dossiers (ancienne structure)
            elif os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    if subitem.endswith(".npy"):
                        subitem_path = os.path.join(item_path, subitem)
                        hand_used = extract_hand_from_filename(subitem)
                        files_with_labels.append((subitem_path, label, hand_used))
    
    return files_with_labels


def load_dataset():
    """
    Charge les données brutes et les normalise via build_feature_vector.
    Produit un dataset cohérent avec le pipeline de prédiction.
    
    Returns:
        tuple: (X, y) - features et labels
    """
    X = []
    y = []
    errors = []

    files_with_labels = find_all_npy_files(DATA_DIR)
    print(f"Fichiers trouves: {len(files_with_labels)}")

    for filepath, label, hand_used in files_with_labels:
        try:
            sample = np.load(filepath, allow_pickle=True).item()

            # Récupération des landmarks bruts
            left_raw = sample.get("left_hand", np.zeros((21, 3)))
            right_raw = sample.get("right_hand", np.zeros((21, 3)))

            # Assurer le bon shape
            if left_raw.shape != (21, 3):
                left_raw = np.zeros((21, 3))
            if right_raw.shape != (21, 3):
                right_raw = np.zeros((21, 3))

            # Pipeline commun NORMALISÉ avec OPTION 1
            # Main active aux positions [0-63], zeros en [63-126] pour lettres
            vec, hand_side, sign_type = build_feature_vector_normalized(
                left_raw, right_raw, hand_used=hand_used
            )

            # Si aucune main détectée → on ignore
            if vec is None:
                errors.append(f"Pas de main: {filepath}")
                continue

            X.append(vec[0])
            y.append(label)

        except Exception as e:
            errors.append(f"Erreur {filepath}: {e}")
            continue

    if errors:
        print(f"\nAvertissements ({len(errors)}):")
        for err in errors[:10]:  # Afficher les 10 premiers
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... et {len(errors) - 10} autres")

    return np.array(X), np.array(y)


def main():
    print("Chargement du dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("Aucune donnee trouvee!")
        print(f"Verifiez que le dossier '{DATA_DIR}' contient des fichiers .npy")
        return

    print(f"\nDataset construit: {X.shape[0]} samples")
    print(f"Shape: {X.shape}")
    print(f"Classes: {len(np.unique(y))}")
    
    # Afficher la distribution
    labels, counts = np.unique(y, return_counts=True)
    print("\nDistribution:")
    for label, count in zip(labels, counts):
        print(f"  {label}: {count} samples")

    # Vérifier les stats pour détecter des anomalies
    print(f"\nStats features:")
    print(f"  Mean: {X.mean():.6f}")
    print(f"  Std: {X.std():.6f}")
    print(f"  Min: {X.min():.6f}")
    print(f"  Max: {X.max():.6f}")

    np.save("X_full.npy", X)
    np.save("y_full.npy", y)

    print("\nDataset sauvegarde: X_full.npy et y_full.npy")


if __name__ == "__main__":
    main()
