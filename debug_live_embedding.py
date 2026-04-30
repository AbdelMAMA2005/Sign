"""
debug_live_embedding.py - Debug des embeddings en temps réel

Compare les embeddings live avec ceux du dataset d'entraînement
pour diagnostiquer les problèmes de cohérence.
"""

import cv2
import numpy as np
import tensorflow as tf
from TrackingModule import holisticDetector
from pipeline_utils import build_feature_vector, debug_feature_vector


def main():
    print("Chargement de l'encodeur...")
    encoder = tf.keras.models.load_model("encoder.h5")
    
    print(f"Encodeur input: {encoder.input_shape}")
    print(f"Encodeur output: {encoder.output_shape}")
    
    # Charger les stats du dataset pour comparaison
    try:
        X_train = np.load("X_full.npy")
        embeddings_train = np.load("embeddings.npy")
        labels_train = np.load("labels.npy")
        
        print(f"\nStats du dataset d'entrainement:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_train mean: {X_train.mean():.6f}")
        print(f"  X_train std: {X_train.std():.6f}")
        print(f"  Embeddings shape: {embeddings_train.shape}")
        print(f"  Embeddings mean: {embeddings_train.mean():.6f}")
        print(f"  Embeddings std: {embeddings_train.std():.6f}")
        
        has_reference = True
    except:
        print("Pas de donnees de reference trouvees")
        has_reference = False

    detector = holisticDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam inaccessible")
        return

    print("\nSPACE = capturer et analyser")
    print("D = debug detaille")
    print("Q = quitter\n")

    captured_embeddings = []
    embedding_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detector.findHolistic(frame, draw=False)
        lm = detector.extract_landmarks()

        left_raw = lm["left_hand"]
        right_raw = lm["right_hand"]

        # Construire le vecteur (même pipeline que train)
        X, hand_side, sign_type = build_feature_vector(left_raw, right_raw)

        # Dessiner les landmarks
        h, w, _ = frame.shape
        for x, y, _ in left_raw:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x*w), int(y*h)), 5, (0, 255, 0), -1)
        for x, y, _ in right_raw:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x*w), int(y*h)), 5, (255, 0, 0), -1)

        # Status
        if X is None:
            status = "Aucune main"
            color = (100, 100, 100)
        else:
            if hand_side == 'left':
                status = "Main GAUCHE"
                color = (0, 255, 0)
            elif hand_side == 'right':
                status = "Main DROITE"
                color = (255, 0, 0)
            else:
                status = "DEUX MAINS"
                color = (0, 255, 255)

        cv2.putText(frame, status, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if X is not None:
            # Afficher les stats du vecteur live
            cv2.putText(frame, f"X mean: {X.mean():.4f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"X std: {X.std():.4f}", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if has_reference:
                # Comparer avec le train
                diff_mean = abs(X.mean() - X_train.mean())
                cv2.putText(frame, f"Diff mean: {diff_mean:.4f}", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(frame, "SPACE=capture | D=debug | Q=quit", 
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Debug Live Embedding", frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # SPACE
            if X is None:
                print("Aucune main detectee!")
            else:
                emb = encoder.predict(X, verbose=0)[0]
                
                print(f"\n=== CAPTURE #{embedding_count} ===")
                print(f"Main: {hand_side}, Type: {sign_type}")
                print(f"X shape: {X.shape}")
                print(f"X mean: {X.mean():.6f}, std: {X.std():.6f}")
                print(f"Embedding shape: {emb.shape}")
                print(f"Embedding mean: {emb.mean():.6f}, std: {emb.std():.6f}")
                
                if has_reference:
                    # Trouver l'embedding le plus proche dans le dataset
                    distances = np.linalg.norm(embeddings_train - emb, axis=1)
                    closest_idx = np.argmin(distances)
                    closest_dist = distances[closest_idx]
                    closest_label = labels_train[closest_idx]
                    
                    print(f"\nPlus proche dans le dataset:")
                    print(f"  Label: {closest_label}")
                    print(f"  Distance: {closest_dist:.4f}")
                    print(f"  Distance moyenne: {distances.mean():.4f}")
                
                np.save(f"live_embedding_{embedding_count}.npy", emb)
                captured_embeddings.append(emb)
                embedding_count += 1
                print(f"Sauvegarde: live_embedding_{embedding_count-1}.npy")
                
        elif key == ord('d'):  # Debug détaillé
            if X is not None:
                print("\n=== DEBUG DETAILLE ===")
                debug_feature_vector(left_raw, right_raw)
                
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n=== RESUME ===")
    print(f"Embeddings captures: {len(captured_embeddings)}")
    
    if len(captured_embeddings) > 0:
        all_emb = np.array(captured_embeddings)
        np.save("live_embeddings_all.npy", all_emb)
        print(f"Sauvegardes dans: live_embeddings_all.npy")


if __name__ == "__main__":
    main()
