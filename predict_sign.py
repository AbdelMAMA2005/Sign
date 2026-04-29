import cv2
import numpy as np
import tensorflow as tf
from TrackingModule import holisticDetector

# --- 90 points du visage (expression uniquement) ---
FACE_INDICES = (
    list(range(70, 108)) +     # Sourcil gauche
    list(range(336, 366)) +    # Sourcil droit
    list(range(33, 63)) +      # Œil gauche (30 points)
    list(range(263, 293)) +    # Œil droit (30 points)
    list(range(0, 18)) +       # Contour bouche
    list(range(61, 81)) +      # Lèvres
    [1, 10, 199]               # Orientation tête
)

# On limite à 90 indices max
FACE_INDICES = FACE_INDICES[:90]

LEFT_SIZE = 21 * 3
RIGHT_SIZE = 21 * 3
FACE_SIZE = 90 * 3
INPUT_SIZE = LEFT_SIZE + RIGHT_SIZE + FACE_SIZE


def draw_hand_points(frame, hand_points, color):
    """Dessine les points de la main."""
    if hand_points.shape != (21, 3):
        return

    h, w, _ = frame.shape
    for x, y, _ in hand_points:
        if x > 0 and y > 0:
            px = int(x * w)
            py = int(y * h)
            cv2.circle(frame, (px, py), 6, color, -1)


def preprocess(left, right, face):
    """Transforme les landmarks en vecteur d'entrée pour le modèle."""
    vec = np.zeros(INPUT_SIZE)

    vec[:LEFT_SIZE] = left.reshape(-1)
    vec[LEFT_SIZE:LEFT_SIZE + RIGHT_SIZE] = right.reshape(-1)
    vec[LEFT_SIZE + RIGHT_SIZE:] = face.reshape(-1)

    return vec.reshape(1, -1)


def main():
    print("📥 Chargement du modèle...")
    model = tf.keras.models.load_model("sign_model.h5")
    labels = np.load("label_encoder.npy")

    detector = holisticDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam inaccessible")
        return

    last_prediction = ""
    stability_counter = 0
    STABILITY_THRESHOLD = 5

    print("🎥 Prêt pour la détection en temps réel !")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        detector.findHolistic(frame, draw=False)
        landmarks = detector.extract_landmarks()

        left = landmarks["left_hand"]
        right = landmarks["right_hand"]
        face_raw = landmarks["face"]

        # Dessin des mains
        draw_hand_points(frame, left, (0, 255, 0))
        draw_hand_points(frame, right, (255, 0, 0))

        # Extraction visage (90 points)
        if face_raw.shape[0] == 468:
            face = face_raw[FACE_INDICES]
        else:
            face = np.zeros((90, 3))

        # Préparation entrée modèle
        X = preprocess(left, right, face)

        # Prédiction
        preds = model.predict(X, verbose=0)
        idx = np.argmax(preds)
        confidence = preds[0][idx]
        prediction = labels[idx]

        # Stabilisation anti-clignotement
        if prediction == last_prediction:
            stability_counter += 1
        else:
            stability_counter = 0

        last_prediction = prediction

        if stability_counter > STABILITY_THRESHOLD:
            display_text = f"{prediction} ({confidence*100:.1f}%)"
        else:
            display_text = "..."

        # Affichage
        cv2.putText(frame, display_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        cv2.putText(frame, "Gauche = Vert | Droite = Bleu",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.imshow("LSF - Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
