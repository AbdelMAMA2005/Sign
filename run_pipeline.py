"""
run_pipeline.py - Script pour exécuter tout le pipeline d'entraînement

Étapes:
1. build_dataset.py - Charge et normalise les données
2. train_full_model.py - Entraîne l'encodeur + classifieur
3. extract_encoder.py - Extrait l'encodeur seul
4. generate_embeddings.py - Génère les embeddings
5. train_classifier.py - Entraîne le classifieur SVM

Après: predict_sign.py pour la prédiction en temps réel
"""

import subprocess
import sys


def run_script(script_name):
    """Exécute un script Python et affiche le résultat"""
    print(f"\n{'='*60}")
    print(f"EXECUTION: {script_name}")
    print('='*60 + '\n')
    
    result = subprocess.run([sys.executable, script_name], 
                          capture_output=False)
    
    if result.returncode != 0:
        print(f"\nERREUR dans {script_name}!")
        return False
    return True


def main():
    print("="*60)
    print("PIPELINE D'ENTRAINEMENT LSF")
    print("="*60)
    
    scripts = [
        "build_dataset.py",
        "train_full_model.py",
        "extract_encoder.py",
        "generate_embeddings.py",
        "train_classifier.py"
    ]
    
    for script in scripts:
        if not run_script(script):
            print(f"\nPipeline interrompu a: {script}")
            return
    
    print("\n" + "="*60)
    print("PIPELINE TERMINE AVEC SUCCES!")
    print("="*60)
    print("\nFichiers crees:")
    print("  - X_full.npy (features normalisees)")
    print("  - y_full.npy (labels)")
    print("  - full_model.h5 (modele complet)")
    print("  - encoder.h5 (encodeur seul)")
    print("  - embeddings.npy (embeddings)")
    print("  - labels.npy (labels)")
    print("  - classifier.pkl (classifieur SVM)")
    print("  - label_encoder.npy (mapping labels)")
    print("\nPour tester:")
    print("  python predict_sign.py")
    print("  python debug_live_embedding.py")


if __name__ == "__main__":
    main()
