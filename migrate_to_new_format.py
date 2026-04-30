"""
migrate_to_new_format.py - Migrer les anciennes données vers le nouveau format

Si vous avez des fichiers comme:
  - data/A/0000.npy
  - data/A/left/0000.npy
  - data/A/right/0000.npy

Ce script les convertit au nouveau format:
  - data/A/A_left_0000.npy
  - data/A/A_right_0000.npy
"""

import os
import shutil
import numpy as np
from pathlib import Path

DATA_DIR = "data"
BACKUP_DIR = "data_backup_old"


def needs_migration(data_dir):
    """Vérifie s'il y a des données à migrer."""
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        
        # Chercher les sous-dossiers (ancienne structure)
        for item in os.listdir(label_path):
            item_path = os.path.join(label_path, item)
            if os.path.isdir(item_path) and item in ("left", "right", "both"):
                return True
        
        # Chercher les fichiers sans nom de main (ancienne structure)
        for item in os.listdir(label_path):
            if item.endswith(".npy") and not any(x in item for x in ["left", "right", "both"]):
                return True
    
    return False


def migrate_old_structure():
    """Migre les anciennes structures vers le nouveau format."""
    
    if not needs_migration(DATA_DIR):
        print("Aucune donnee a migrer. Vous utilisez deja le nouveau format.")
        return
    
    print(f"Migration detects. Creation de la sauvegarde dans {BACKUP_DIR}/")
    
    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(DATA_DIR, BACKUP_DIR)
    print("Sauvegarde creee!")
    
    migrated = 0
    
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue
        
        # Traiter les sous-dossiers (left, right, both)
        for hand_dir in ("left", "right", "both"):
            hand_path = os.path.join(label_path, hand_dir)
            if os.path.isdir(hand_path):
                print(f"Migration de {label}/{hand_dir}/ ...")
                
                for idx, filename in enumerate(sorted(os.listdir(hand_path))):
                    if filename.endswith(".npy"):
                        old_path = os.path.join(hand_path, filename)
                        
                        # Nouveau nom: LABEL_hand_INDEX.npy
                        new_filename = f"{label}_{hand_dir}_{idx:04d}.npy"
                        new_path = os.path.join(label_path, new_filename)
                        
                        # Charger et mettre a jour si necessaire
                        sample = np.load(old_path, allow_pickle=True).item()
                        sample["hand_used"] = hand_dir
                        
                        # Sauvegarder au nouveau lieu
                        np.save(new_path, sample)
                        print(f"  {filename} → {new_filename}")
                        migrated += 1
                
                # Supprimer le dossier old
                shutil.rmtree(hand_path)
                print(f"Dossier {hand_dir}/ supprime")
        
        # Traiter les fichiers directs dans data/LABEL/ sans indicateur de main
        for filename in list(os.listdir(label_path)):
            filepath = os.path.join(label_path, filename)
            if os.path.isfile(filepath) and filename.endswith(".npy"):
                # Verifier si le nom contient deja _left, _right, ou _both
                if not any(x in filename for x in ["_left", "_right", "_both"]):
                    print(f"Fichier ambigu detecte: {filename}")
                    print("  Ce fichier n'a pas d'indicateur de main (_left/_right/_both)")
                    print("  Veuillez le renommer manuellement ou utiliser le mode sans indicateur")
    
    print(f"\nMigration terminees! {migrated} fichiers migres")
    print("Vous pouvez supprimer le dossier de sauvegarde si tout est ok:")
    print(f"  rm -rf {BACKUP_DIR}")


if __name__ == "__main__":
    migrate_old_structure()
