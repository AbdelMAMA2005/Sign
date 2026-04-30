"""
data_collection.py - Collecte des données LSF

CORRECTION: Structure de dossiers simplifiée.
Les fichiers sont maintenant sauvegardés dans data/{label}/
au lieu de data/{label}/{hand_folder}/

Le mode de collection (gauche/droite/deux mains) est encodé
dans les données elles-mêmes, pas dans la structure des dossiers.
"""

import cv2
import numpy as np
import os
import time
from TrackingModule import holisticDetector


def is_letter(label: str) -> bool:
    """Vérifie si le label est une lettre (une seule main)"""
    return len(label) == 1 and label.isalpha()


def draw_hand_points(frame, hand_points, color):
    """Dessine les points de la main sur le frame"""
    if hand_points.shape != (21, 3):
        return

    h, w, _ = frame.shape
    for x, y, _ in hand_points:
        if x > 0 and y > 0:
            px = int(x * w)
            py = int(y * h)
            cv2.circle(frame, (px, py), 6, color, -1)


def detection_ok(mode, left_raw, right_raw):
    """
    Vérifie si la détection correspond au mode souhaité.
    
    CORRECTION: Le visage n'est plus vérifié car il est
    complètement exclu du pipeline.
    
    Args:
        mode: "1" (gauche), "2" (droite), "3" (deux mains)
        left_raw: landmarks main gauche
        right_raw: landmarks main droite
        
    Returns:
        tuple (bool, str): (ok, reason)
    """
    left_ok = np.abs(left_raw).sum() > 1e-6
    right_ok = np.abs(right_raw).sum() > 1e-6

    # Mode 1 = main gauche SEULE obligatoire
    if mode == "1":
        if not left_ok:
            return False, "Main gauche non detectee"
        if right_ok:
            return False, "Retirez la main droite"
        return True, ""

    # Mode 2 = main droite SEULE obligatoire
    if mode == "2":
        if not right_ok:
            return False, "Main droite non detectee"
        if left_ok:
            return False, "Retirez la main gauche"
        return True, ""

    # Mode 3 = DEUX mains obligatoires
    if mode == "3":
        if not left_ok:
            return False, "Main gauche manquante"
        if not right_ok:
            return False, "Main droite manquante"
        return True, ""

    return False, "Mode inconnu"


def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    detector = holisticDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam inaccessible")
        return

    print("\n=== MODE DE COLLECTE ===")
    print("1 = Main gauche (lettres)")
    print("2 = Main droite (lettres)")
    print("3 = Deux mains (mots)")
    mode = input("Choisissez le mode (1/2/3) : ").strip()

    if mode not in ("1", "2", "3"):
        print("Mode invalide")
        return

    label = input("Entrez le label (lettre ou mot) : ").strip().upper()
    if not label:
        print("Label invalide")
        return

    # Créer le dossier pour ce label
    # CORRECTION: Structure plate data/{label}/ sans sous-dossiers
    label_dir = os.path.join(data_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    # Determiner le suffixe de main pour ce mode
    hand_suffix = "left" if mode == "1" else "right" if mode == "2" else "both"

    # Compter les fichiers existants POUR CETTE MAIN SEULEMENT
    # Exemple: si on a A_left_0000.npy et A_left_0001.npy, et on veut enregistrer A_right,
    # on compte uniquement les fichiers A_right_*.npy
    existing_files = [f for f in os.listdir(label_dir) 
                      if f.endswith(".npy") and f"_{hand_suffix}_" in f]
    default_start = len(existing_files)

    print(f"\nFichiers existants pour {label}_{hand_suffix}_: {default_start}")
    start_input = input(f"Commencer a quel numero? (defaut: {default_start}) : ").strip()
    
    if start_input == "":
        start_count = default_start
    else:
        try:
            start_count = int(start_input)
        except ValueError:
            print("Numero invalide, utilisation du defaut")
            start_count = default_start

    collecting = False
    sample_count = start_count
    max_samples = start_count + 100
    last_save = 0
    interval = 0.5

    print(f"\nDossier: {label_dir}")
    print(f"Fichiers existants: {start_count}")
    print("C = commencer | S = pause | Q = quitter\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detector.findHolistic(frame, draw=False)
        lm = detector.extract_landmarks()

        left_raw = lm["left_hand"]
        right_raw = lm["right_hand"]

        # Dessin des landmarks
        if mode in ("1", "3"):
            draw_hand_points(frame, left_raw, (0, 255, 0))  # Vert = gauche
        if mode in ("2", "3"):
            draw_hand_points(frame, right_raw, (255, 0, 0))  # Bleu = droite

        # Collecte
        if collecting and time.time() - last_save >= interval:
            ok, reason = detection_ok(mode, left_raw, right_raw)

            if not ok:
                cv2.putText(frame, f"! {reason}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Préparer les données à sauvegarder
                # IMPORTANT: On sauvegarde les landmarks BRUTS
                # La normalisation se fait au moment de l'entraînement
                
                if mode == "1":  # Main gauche seule
                    left_save = left_raw.copy()
                    right_save = np.zeros((21, 3))
                elif mode == "2":  # Main droite seule
                    left_save = np.zeros((21, 3))
                    right_save = right_raw.copy()
                else:  # Mode 3: deux mains
                    left_save = left_raw.copy()
                    right_save = right_raw.copy()

                sample = {
                    "left_hand": left_save,
                    "right_hand": right_save,
                    "hand_used": hand_suffix,  # "left", "right", ou "both"
                    "mode": mode,  # Utile pour debug
                    "label": label
                }

                # Format: LABEL_HAND_INDEX.npy (ex: A_left_0000.npy)
                filename = f"{label}_{hand_suffix}_{sample_count:04d}.npy"
                filepath = os.path.join(label_dir, filename)
                np.save(filepath, sample)

                print(f"Sample {sample_count}: {filename}")
                sample_count += 1
                last_save = time.time()

                if sample_count >= max_samples:
                    print(f"\nCollecte terminee: {sample_count - start_count} nouveaux samples")
                    collecting = False

        # UI
        status = "ENREGISTREMENT" if collecting else "PAUSE"
        color = (0, 255, 0) if collecting else (0, 165, 255)
        cv2.putText(frame, f"{label} | {status} | {sample_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        mode_text = {
            "1": "Mode: GAUCHE",
            "2": "Mode: DROITE", 
            "3": "Mode: DEUX MAINS"
        }
        cv2.putText(frame, mode_text[mode], (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Collecte LSF", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            collecting = True
            print("Collecte demarree...")
        elif key == ord('s'):
            collecting = False
            print("Pause...")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal samples pour {label}: {sample_count}")


if __name__ == "__main__":
    main()
