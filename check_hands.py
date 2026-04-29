import os
import numpy as np

def hand_present(hand_array):
    """Retourne True si la main contient des valeurs non nulles."""
    return np.sum(hand_array) != 0


def check_file(path):
    """Analyse un fichier .npy et retourne un résumé."""
    data = np.load(path, allow_pickle=True).item()

    # Certains fichiers (face) n'ont pas de mains
    if "left_hand" not in data or "right_hand" not in data:
        return "Fichier visage (pas de mains)"

    left = data["left_hand"]
    right = data["right_hand"]

    left_ok = hand_present(left)
    right_ok = hand_present(right)

    if left_ok and right_ok:
        status = "Deux mains détectées"
    elif left_ok:
        status = "Main gauche uniquement"
    elif right_ok:
        status = "Main droite uniquement"
    else:
        status = "❌ Aucune main détectée"

    return status


def main():
    root = "data"
    print("\n=== Vérification des fichiers .npy ===\n")

    for label in os.listdir(root):
        label_path = os.path.join(root, label)
        if not os.path.isdir(label_path):
            continue

        print(f"\n--- Label : {label} ---")

        # Parcourt left / right / both / face
        for sub in os.listdir(label_path):
            sub_path = os.path.join(label_path, sub)
            if not os.path.isdir(sub_path):
                continue

            print(f"\n  > Sous-dossier : {sub}")

            for file in os.listdir(sub_path):
                if file.endswith(".npy"):
                    path = os.path.join(sub_path, file)
                    status = check_file(path)
                    print(f"    {file} → {status}")


if __name__ == "__main__":
    main()
