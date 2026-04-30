"""
check_consistency.py - Vérifier la cohérence du pipeline OPTION 1

Cet outil vérifie:
1. Les fichiers ont les bons noms (LABEL_hand_INDEX.npy)
2. Le contenu des fichiers est cohérent
3. La main active est à la bonne place dans le vecteur normalisé
"""

import os
import re
import numpy as np
from pipeline_utils import build_feature_vector_normalized

DATA_DIR = "data"


def check_filenames():
    """Vérifie que tous les fichiers suivent la convention de noms."""
    print("=" * 60)
    print("1. VERIFICATION DES NOMS DE FICHIERS")
    print("=" * 60)
    
    pattern = r'^([A-Z]+)_(left|right|both)_(\d{4})\.npy$'
    issues = []
    total_files = 0
    
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue
        
        for filename in os.listdir(label_path):
            if filename.endswith(".npy"):
                total_files += 1
                
                if not re.match(pattern, filename):
                    issues.append(f"  ❌ {label}/{filename}")
                    print(f"  ❌ Format incorrect: {filename}")
                    print(f"     Esperé: {label}_(left|right|both)_XXXX.npy")
    
    if not issues:
        print(f"✓ Tous les {total_files} fichiers ont le bon format!")
    else:
        print(f"\n⚠️  {len(issues)} fichier(s) avec format incorrect")
    
    return len(issues) == 0


def check_file_content():
    """Vérifie le contenu des fichiers .npy."""
    print("\n" + "=" * 60)
    print("2. VERIFICATION DU CONTENU DES FICHIERS")
    print("=" * 60)
    
    issues = []
    checked = 0
    
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue
        
        for filename in os.listdir(label_path):
            if not filename.endswith(".npy"):
                continue
            
            filepath = os.path.join(label_path, filename)
            checked += 1
            
            try:
                sample = np.load(filepath, allow_pickle=True).item()
                
                # Verifier les clés
                required_keys = {"left_hand", "right_hand", "hand_used", "label"}
                if not required_keys.issubset(sample.keys()):
                    missing = required_keys - set(sample.keys())
                    issues.append(f"  ❌ {filename}: Cles manquantes: {missing}")
                    print(f"  ❌ {filename}: Cles manquantes: {missing}")
                    continue
                
                # Verifier les shapes
                left = sample["left_hand"]
                right = sample["right_hand"]
                hand_used = sample["hand_used"]
                
                if left.shape != (21, 3):
                    issues.append(f"  ❌ {filename}: left_hand shape incorrect {left.shape}")
                
                if right.shape != (21, 3):
                    issues.append(f"  ❌ {filename}: right_hand shape incorrect {right.shape}")
                
                # Verifier la coherence: hand_used vs contenu
                left_active = np.abs(left).sum() > 1e-6
                right_active = np.abs(right).sum() > 1e-6
                
                if hand_used == "left" and not left_active:
                    issues.append(f"  ❌ {filename}: hand_used='left' mais left_hand est vide")
                    print(f"  ❌ {filename}: hand_used='left' mais left_hand est vide")
                
                if hand_used == "left" and right_active:
                    issues.append(f"  ⚠️  {filename}: hand_used='left' mais right_hand n'est pas vide")
                    print(f"  ⚠️  {filename}: hand_used='left' mais right_hand n'est pas vide")
                
                if hand_used == "right" and not right_active:
                    issues.append(f"  ❌ {filename}: hand_used='right' mais right_hand est vide")
                    print(f"  ❌ {filename}: hand_used='right' mais right_hand est vide")
                
                if hand_used == "right" and left_active:
                    issues.append(f"  ⚠️  {filename}: hand_used='right' mais left_hand n'est pas vide")
                    print(f"  ⚠️  {filename}: hand_used='right' mais left_hand n'est pas vide")
                
                if hand_used == "both" and (not left_active or not right_active):
                    issues.append(f"  ❌ {filename}: hand_used='both' mais une main est vide")
                    print(f"  ❌ {filename}: hand_used='both' mais une main est vide")
                
            except Exception as e:
                issues.append(f"  ❌ {filename}: Erreur lecture - {e}")
                print(f"  ❌ {filename}: Erreur lecture - {e}")
    
    if not issues:
        print(f"✓ Tous les {checked} fichiers sont coherents!")
    else:
        print(f"\n⚠️  {len(issues)} probleme(s) detecte(s)")
    
    return len(issues) == 0


def check_vector_consistency():
    """Vérifie que la normalisation place la main active aux bonnes positions."""
    print("\n" + "=" * 60)
    print("3. VERIFICATION DE LA NORMALISATION DU VECTEUR")
    print("=" * 60)
    
    issues = []
    checked = 0
    
    for label in sorted(os.listdir(DATA_DIR)):
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue
        
        files_by_hand = {"left": [], "right": [], "both": []}
        
        for filename in sorted(os.listdir(label_path)):
            if not filename.endswith(".npy"):
                continue
            
            filepath = os.path.join(label_path, filename)
            sample = np.load(filepath, allow_pickle=True).item()
            hand_used = sample.get("hand_used")
            
            if hand_used in files_by_hand:
                files_by_hand[hand_used].append((filepath, filename))
        
        # Verifier que pour la même lettre, left et right produisent le même vecteur
        # (positions 0-63 doivent être différents, mais la structure doit être cohérente)
        
        if files_by_hand["left"] and files_by_hand["right"]:
            print(f"\nLabel: {label}")
            
            # Charger un exemple gauche et droite
            left_path, left_name = files_by_hand["left"][0]
            right_path, right_name = files_by_hand["right"][0]
            
            left_sample = np.load(left_path, allow_pickle=True).item()
            right_sample = np.load(right_path, allow_pickle=True).item()
            
            # Normaliser
            vec_left, hand_side_left, sign_type_left = build_feature_vector_normalized(
                left_sample["left_hand"], 
                left_sample["right_hand"], 
                hand_used="left"
            )
            
            vec_right, hand_side_right, sign_type_right = build_feature_vector_normalized(
                right_sample["left_hand"], 
                right_sample["right_hand"], 
                hand_used="right"
            )
            
            print(f"  Left ({left_name}):")
            print(f"    - Positions 0-62 (main): sum = {np.abs(vec_left[0, :63]).sum():.6f}")
            print(f"    - Positions 63-125 (zeros): sum = {np.abs(vec_left[0, 63:]).sum():.6f}")
            
            print(f"  Right ({right_name}):")
            print(f"    - Positions 0-62 (main): sum = {np.abs(vec_right[0, :63]).sum():.6f}")
            print(f"    - Positions 63-125 (zeros): sum = {np.abs(vec_right[0, 63:]).sum():.6f}")
            
            # Verifier que les deux ont la même structure (positions 0-62 actif, 63-125 zeros)
            if np.abs(vec_left[0, :63]).sum() > 1e-6 and np.abs(vec_left[0, 63:]).sum() < 1e-6:
                if np.abs(vec_right[0, :63]).sum() > 1e-6 and np.abs(vec_right[0, 63:]).sum() < 1e-6:
                    print(f"  ✓ Structure coherente (OPTION 1)")
                else:
                    print(f"  ❌ Right n'a pas la bonne structure!")
                    issues.append(f"  ❌ {label}: Right vector n'a pas zeros en positions 63-125")
            else:
                print(f"  ❌ Left n'a pas la bonne structure!")
                issues.append(f"  ❌ {label}: Left vector n'a pas la structure attendue")
            
            checked += 1
    
    if not issues:
        print(f"\n✓ {checked} comparaisons effectuees, structure OK!")
    else:
        print(f"\n⚠️  {len(issues)} probleme(s) de structure")
    
    return len(issues) == 0


def main():
    print("\n" + "=" * 60)
    print("VERIFICATION DE LA COHERENCE - PIPELINE OPTION 1")
    print("=" * 60 + "\n")
    
    check1 = check_filenames()
    check2 = check_file_content()
    check3 = check_vector_consistency()
    
    print("\n" + "=" * 60)
    print("RESUME")
    print("=" * 60)
    
    if check1 and check2 and check3:
        print("✓ TOUT EST BON! Pipeline coherent et pret.")
    else:
        print("⚠️  Des problemes ont ete detectes. Voir les details ci-dessus.")


if __name__ == "__main__":
    main()
