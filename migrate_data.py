"""
migrate_data.py - Migration des anciennes données

Si vous aviez l'ancienne structure:
  data/{label}/{hand}/*.npy

Ce script les déplace vers la nouvelle structure:
  data/{label}/*.npy

NOTE: Le nouveau build_dataset.py gère les deux structures,
donc cette migration est optionnelle.
"""

import os
import shutil


def main():
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print("Dossier 'data' non trouve!")
        return
    
    files_moved = 0
    
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        
        # Chercher les sous-dossiers (left, right, both)
        for subfolder in os.listdir(label_path):
            subfolder_path = os.path.join(label_path, subfolder)
            
            if not os.path.isdir(subfolder_path):
                continue
            
            # Déplacer les fichiers .npy vers le dossier parent
            for filename in os.listdir(subfolder_path):
                if not filename.endswith(".npy"):
                    continue
                
                src = os.path.join(subfolder_path, filename)
                
                # Créer un nouveau nom unique
                new_filename = f"{label}_{subfolder}_{filename}"
                dst = os.path.join(label_path, new_filename)
                
                # Éviter les doublons
                counter = 0
                while os.path.exists(dst):
                    counter += 1
                    new_filename = f"{label}_{subfolder}_{counter:04d}.npy"
                    dst = os.path.join(label_path, new_filename)
                
                shutil.move(src, dst)
                print(f"Deplace: {src} -> {dst}")
                files_moved += 1
            
            # Supprimer le sous-dossier vide
            try:
                os.rmdir(subfolder_path)
                print(f"Supprime dossier vide: {subfolder_path}")
            except OSError:
                pass  # Non vide, on le laisse
    
    print(f"\nMigration terminee: {files_moved} fichiers deplaces")


if __name__ == "__main__":
    confirm = input("Cette operation va reorganiser vos donnees. Continuer? (o/n): ")
    if confirm.lower() == 'o':
        main()
    else:
        print("Migration annulee.")
