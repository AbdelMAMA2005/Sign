"""
predict_sign.py - Prédiction en temps réel des signes LSF

IMPORTANT: Utilise EXACTEMENT le même pipeline de normalisation
que l'entraînement (via build_feature_vector).
"""

import cv2
import numpy as np
import tensorflow as tf
import joblib
from TrackingModule import holisticDetector
from pipeline_utils import build_feature_vector_normalized


def draw_hand_points(frame, hand_points, color):
    """Dessine les points des mains sur le frame"""
    if hand_points.shape != (21, 3):
        return

    h, w, _ = frame.shape
    for x, y, _ in hand_points:
        if x > 0 and y > 0:
            px = int(x * w)
            py = int(y * h)
            cv2.circle(frame, (px, py), 6, color, -1)


def format_prediction_text(hand_side, sign_type, label, confidence):
    """
    Formate le texte de prédiction.
    
    Exemples:
    - "main gauche, lettre A (92.3%)"
    - "main droite, lettre C (87.5%)"
    - "deux mains, mot: MERCI (94.1%)"
    """
    if hand_side == 'left':
        hand_str = "main gauche"
    elif hand_side == 'right':
        hand_str = "main droite"
    elif hand_side == 'both':
        hand_str = "deux mains"
    else:
        return ""
    
    if sign_type == 'letter':
        return f"{hand_str}, lettre {label} ({confidence*100:.1f}%)"
    elif sign_type == 'word':
        return f"{hand_str}, mot: {label} ({confidence*100:.1f}%)"
    else:
        return f"{hand_str}: {label} ({confidence*100:.1f}%)"


def main():
    print("Chargement de l'encodeur et du classifieur...")

    try:
        encoder = tf.keras.models.load_model("encoder.h5")
        clf = joblib.load("classifier.pkl")
        labels = np.load("label_encoder.npy")
    except Exception as e:
        print(f"Erreur chargement modeles: {e}")
        print("Avez-vous execute le pipeline d'entrainement?")
        return

    print(f"Encodeur input: {encoder.input_shape}")
    print(f"Labels: {labels}")

    detector = holisticDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam inaccessible")
        return

    last_prediction = ""
    last_hand_side = None
    stability_counter = 0
    STABILITY_THRESHOLD = 5

    print("\nPret pour la detection!")
    print("  - Une main = lettre")
    print("  - Deux mains = mot")
    print("  Q = quitter\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detector.findHolistic(frame, draw=False)
        landmarks = detector.extract_landmarks()

        left_raw = landmarks["left_hand"]
        right_raw = landmarks["right_hand"]

        # Dessin des landmarks BRUTS (avant normalisation)
        draw_hand_points(frame, left_raw, (0, 255, 0))   # Vert = gauche
        draw_hand_points(frame, right_raw, (255, 0, 0))  # Bleu = droite

        # Construction du vecteur (AVEC normalisation - OPTION 1)
        # IMPORTANT: C'est EXACTEMENT le même pipeline que l'entraînement
        # hand_used=None → auto-détection basée sur la présence des mains
        X, hand_side, sign_type = build_feature_vector_normalized(left_raw, right_raw, hand_used=None)

        # Aucune main détectée
        if X is None:
            cv2.putText(frame, "Aucune main detectee...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            cv2.putText(frame, "Vert=Gauche | Bleu=Droite", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow("LSF - Prediction", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Embedding via l'encodeur
        embedding = encoder.predict(X, verbose=0)

        # Classification
        proba = clf.predict_proba(embedding)[0]
        idx = np.argmax(proba)
        prediction = labels[idx]
        confidence = proba[idx]

        # Stabilisation
        if prediction == last_prediction and hand_side == last_hand_side:
            stability_counter += 1
        else:
            stability_counter = 0

        last_prediction = prediction
        last_hand_side = hand_side

        # Affichage
        if stability_counter > STABILITY_THRESHOLD:
            display_text = format_prediction_text(
                hand_side, sign_type, prediction, confidence
            )
            text_color = (0, 255, 0)  # Vert quand stable
        else:
            display_text = "Stabilisation..."
            text_color = (0, 165, 255)  # Orange

        cv2.putText(frame, display_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

        # Info sur la main
        if hand_side == 'left':
            hand_info = "Main GAUCHE"
            hand_color = (0, 255, 0)
        elif hand_side == 'right':
            hand_info = "Main DROITE"
            hand_color = (255, 0, 0)
        else:
            hand_info = "DEUX MAINS"
            hand_color = (0, 255, 255)

        cv2.putText(frame, hand_info, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, hand_color, 2)

        # Top 3 prédictions
        top_indices = np.argsort(proba)[-3:][::-1]
        y_offset = 120
        for i, idx in enumerate(top_indices):
            label_text = f"{i+1}. {labels[idx]}: {proba[idx]*100:.1f}%"
            cv2.putText(frame, label_text, (10, y_offset + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(frame, "(Q pour quitter)", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("LSF - Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Prediction terminee")


if __name__ == "__main__":
    main()
