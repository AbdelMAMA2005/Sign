import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

DATA_DIR = "data"

# Dimensions fixes
LEFT_SIZE = 21 * 3
RIGHT_SIZE = 21 * 3
FACE_SIZE = 90 * 3   # car tu utilises ~90 points du visage

INPUT_SIZE = LEFT_SIZE + RIGHT_SIZE + FACE_SIZE


def load_sample(path, is_word):
    """Charge un fichier .npy et renvoie un vecteur normalisé."""
    data = np.load(path, allow_pickle=True).item()

    left = data.get("left_hand", np.zeros((21, 3))).reshape(-1)
    right = data.get("right_hand", np.zeros((21, 3))).reshape(-1)

    if is_word:
        face = data.reshape(-1) if isinstance(data, np.ndarray) else np.zeros(FACE_SIZE)
    else:
        face = np.zeros(FACE_SIZE)

    # Concaténation
    vec = np.zeros(INPUT_SIZE)
    vec[:LEFT_SIZE] = left
    vec[LEFT_SIZE:LEFT_SIZE + RIGHT_SIZE] = right
    vec[LEFT_SIZE + RIGHT_SIZE:] = face

    return vec


def load_dataset():
    X = []
    y = []

    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue

        is_word = not (len(label) == 1 and label.isalpha())

        # Sous-dossiers : left / right / both
        for sub in ["left", "right", "both"]:
            sub_path = os.path.join(label_path, sub)
            if not os.path.isdir(sub_path):
                continue

            for file in os.listdir(sub_path):
                if file.endswith(".npy"):
                    path = os.path.join(sub_path, file)
                    sample = np.load(path, allow_pickle=True).item()

                    left = sample["left_hand"].reshape(-1)
                    right = sample["right_hand"].reshape(-1)

                    # Face pour les mots
                    if is_word:
                        face_path = os.path.join(label_path, "face", file.replace(sub, "face"))
                        if os.path.exists(face_path):
                            face = np.load(face_path)
                            face = face.reshape(-1)
                        else:
                            face = np.zeros(FACE_SIZE)
                    else:
                        face = np.zeros(FACE_SIZE)

                    # Concaténation
                    vec = np.zeros(INPUT_SIZE)
                    vec[:LEFT_SIZE] = left
                    vec[LEFT_SIZE:LEFT_SIZE + RIGHT_SIZE] = right
                    vec[LEFT_SIZE + RIGHT_SIZE:] = face

                    X.append(vec)
                    y.append(label)

    return np.array(X), np.array(y)


def build_model(num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(INPUT_SIZE,)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    print("📥 Chargement du dataset...")
    X, y = load_dataset()

    print("Dataset chargé :", X.shape, "samples")

    # Encodage des labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, shuffle=True, stratify=y_cat
    )

    print("📦 Entraînement du modèle...")
    model = build_model(num_classes=y_cat.shape[1])

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32
    )

    print("💾 Sauvegarde du modèle...")
    model.save("sign_model.h5")
    np.save("label_encoder.npy", encoder.classes_)

    print("🎉 Entraînement terminé !")


if __name__ == "__main__":
    main()
