import cv2
import numpy as np
import os
import time
from TrackingModule import holisticDetector

# --- Indices utiles du visage (expression uniquement) ---
FACE_INDICES = (
    list(range(70, 108)) +
    list(range(336, 366)) +
    list(range(33, 134)) +
    list(range(362, 264, -1)) +
    list(range(0, 18)) +
    list(range(61, 292)) +
    [1, 10, 199]
)

def is_letter(label: str) -> bool:
    """Retourne True si le label est une lettre A-Z."""
    return len(label) == 1 and label.isalpha()

def draw_hand_points(frame, hand_points, color):
    """Dessine les points de la main en convertissant les coords normalisées."""
    if hand_points.shape != (21, 3):
        return

    h, w, _ = frame.shape

    for x, y, _ in hand_points:
        if x > 0 and y > 0:
            px = int(x * w)
            py = int(y * h)
            cv2.circle(frame, (px, py), 6, color, -1)


def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    detector = holisticDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam inaccessible")
        return

    print("\n=== MODE DE COLLECTE ===")
    print("1 = Main gauche")
    print("2 = Main droite")
    print("3 = Deux mains")
    mode = input("Choisissez le mode (1/2/3) : ").strip()

    if mode not in ("1", "2", "3"):
        print("❌ Mode invalide")
        return

    label = input("Entrez le label (lettre ou mot) : ").strip().upper()
    if not label:
        print("❌ Label invalide")
        return

    # Lettres = pas d'expression faciale
    use_face = not is_letter(label)

    collecting = False
    sample_count = 0
    max_samples = 100
    last_save = 0
    interval = 0.5

    print("\nC = commencer | S = pause | Q = quitter\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detector.findHolistic(frame, draw=False)
        landmarks = detector.extract_landmarks()

        # --- Extraction sécurisée ---
        left_hand = landmarks["left_hand"]
        right_hand = landmarks["right_hand"]
        face_raw = landmarks["face"]

        # --- Affichage des mains ---
        if mode in ("1", "3"):
            draw_hand_points(frame, left_hand, (0, 255, 0))   # VERT = gauche
        if mode in ("2", "3"):
            draw_hand_points(frame, right_hand, (255, 0, 0))  # BLEU = droite

        # --- Collecte ---
        if collecting and time.time() - last_save >= interval:

            # Copie pour la sauvegarde
            left_save = left_hand.copy()
            right_save = right_hand.copy()

            # Mode de collecte
            if mode == "1":
                right_save = np.zeros((21, 3))
                hand_folder = "left"
            elif mode == "2":
                left_save = np.zeros((21, 3))
                hand_folder = "right"
            else:
                hand_folder = "both"

            # Création du dossier mains
            hand_dir = os.path.join("data", label, hand_folder)
            os.makedirs(hand_dir, exist_ok=True)

            # Expression faciale uniquement pour les mots
            if use_face and face_raw.shape[0] == 468:
                face = face_raw[FACE_INDICES]
            else:
                face = np.zeros((len(FACE_INDICES), 3))

            # --- Sauvegarde des mains ---
            hand_filename = f"{label}_{hand_folder}_{sample_count:03d}.npy"
            np.save(os.path.join(hand_dir, hand_filename), {
                "left_hand": left_save,
                "right_hand": right_save
            })

            print(f"✔ Mains sauvegardées : {hand_filename}")

            # --- Sauvegarde du visage (uniquement pour les mots) ---
            if use_face:
                face_dir = os.path.join("data", label, "face")
                os.makedirs(face_dir, exist_ok=True)

                face_filename = f"{label}_face_{sample_count:03d}.npy"
                np.save(os.path.join(face_dir, face_filename), face)

                print(f"✔ Visage sauvegardé : {face_filename}")

            sample_count += 1
            last_save = time.time()

            if sample_count >= max_samples:
                print("✅ Collecte terminée")
                collecting = False

        # --- UI ---
        status = "ENREGISTREMENT" if collecting else "PAUSE"
        cv2.putText(frame, f"{label} | {status}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, "Gauche = Vert | Droite = Bleu",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.imshow("Collecte LSF", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            collecting = True
        elif key == ord('s'):
            collecting = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
