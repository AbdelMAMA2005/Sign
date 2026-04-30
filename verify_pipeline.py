"""
verify_pipeline.py - Vérifie la cohérence du pipeline

Ce script vérifie que le pipeline de normalisation est identique
entre l'entraînement et la prédiction live.
"""

import numpy as np
import os
from pipeline_utils import build_feature_vector, INPUT_SIZE


def verify_data_files():
    """Vérifie les fichiers de données"""
    print("=== VERIFICATION DES DONNEES ===\n")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("ERREUR: Dossier 'data' non trouve!")
        return False
    
    # Compter les fichiers
    total_files = 0
    labels_found = {}
    
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        
        count = 0
        for item in os.listdir(label_path):
            item_path = os.path.join(label_path, item)
            
            if item.endswith(".npy"):
                count += 1
            elif os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    if subitem.endswith(".npy"):
                        count += 1
        
        if count > 0:
            labels_found[label] = count
            total_files += count
    
    print(f"Labels trouves: {len(labels_found)}")
    for label, count in sorted(labels_found.items()):
        print(f"  {label}: {count} fichiers")
    print(f"Total: {total_files} fichiers")
    
    return total_files > 0


def verify_normalization():
    """Vérifie que la normalisation est cohérente"""
    print("\n=== VERIFICATION NORMALISATION ===\n")
    
    # Créer des données de test
    test_hand = np.random.rand(21, 3).astype(np.float32)
    test_hand[0] = [0.5, 0.5, 0.0]  # Poignet au centre
    
    # Normaliser plusieurs fois
    results = []
    for _ in range(5):
        vec, _, _ = build_feature_vector(test_hand, np.zeros((21, 3)))
        if vec is not None:
            results.append(vec[0])
    
    if len(results) < 2:
        print("ERREUR: Normalisation echoue!")
        return False
    
    # Vérifier que tous les résultats sont identiques
    all_same = all(np.allclose(results[0], r) for r in results[1:])
    
    if all_same:
        print("OK: Normalisation deterministe")
        print(f"  Output shape: {results[0].shape}")
        print(f"  Expected: ({INPUT_SIZE},)")
    else:
        print("ERREUR: Normalisation non deterministe!")
        return False
    
    return True


def verify_sample_processing():
    """Vérifie le traitement d'un sample réel"""
    print("\n=== VERIFICATION SAMPLE REEL ===\n")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        return False
    
    # Trouver un fichier sample
    sample_path = None
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        
        for item in os.listdir(label_path):
            if item.endswith(".npy"):
                sample_path = os.path.join(label_path, item)
                break
        
        if sample_path:
            break
    
    if not sample_path:
        print("Aucun sample trouve")
        return False
    
    print(f"Test avec: {sample_path}")
    
    try:
        sample = np.load(sample_path, allow_pickle=True).item()
        
        left_raw = sample.get("left_hand", np.zeros((21, 3)))
        right_raw = sample.get("right_hand", np.zeros((21, 3)))
        
        print(f"  Left shape: {left_raw.shape}")
        print(f"  Right shape: {right_raw.shape}")
        print(f"  Left sum: {left_raw.sum():.4f}")
        print(f"  Right sum: {right_raw.sum():.4f}")
        
        vec, hand_side, sign_type = build_feature_vector(left_raw, right_raw)
        
        if vec is None:
            print("  Resultat: Aucune main detectee")
        else:
            print(f"  Resultat: {hand_side}, {sign_type}")
            print(f"  Vec shape: {vec.shape}")
            print(f"  Vec mean: {vec.mean():.6f}")
            print(f"  Vec std: {vec.std():.6f}")
        
        return True
        
    except Exception as e:
        print(f"ERREUR: {e}")
        return False


def verify_model_files():
    """Vérifie les fichiers de modèle"""
    print("\n=== VERIFICATION MODELES ===\n")
    
    files = [
        ("X_full.npy", "Features"),
        ("y_full.npy", "Labels"),
        ("full_model.h5", "Modele complet"),
        ("encoder.h5", "Encodeur"),
        ("embeddings.npy", "Embeddings"),
        ("labels.npy", "Labels embeddings"),
        ("classifier.pkl", "Classifieur"),
        ("label_encoder.npy", "Label encoder")
    ]
    
    all_ok = True
    for filename, description in files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"OK: {filename} ({description}) - {size/1024:.1f} KB")
        else:
            print(f"MANQUANT: {filename} ({description})")
            all_ok = False
    
    return all_ok


def main():
    print("="*60)
    print("VERIFICATION DU PIPELINE LSF")
    print("="*60)
    
    results = []
    
    results.append(("Donnees", verify_data_files()))
    results.append(("Normalisation", verify_normalization()))
    results.append(("Sample", verify_sample_processing()))
    results.append(("Modeles", verify_model_files()))
    
    print("\n" + "="*60)
    print("RESUME")
    print("="*60 + "\n")
    
    all_ok = True
    for name, ok in results:
        status = "OK" if ok else "ERREUR"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False
    
    print()
    if all_ok:
        print("Tout est bon! Vous pouvez utiliser predict_sign.py")
    else:
        print("Des problemes ont ete detectes. Verifiez les erreurs ci-dessus.")


if __name__ == "__main__":
    main()
