import os
import numpy as np
import string

# Indices utiles du visage
FACE_INDICES = (
    list(range(70, 108)) +
    list(range(336, 366)) +
    list(range(33, 134)) +
    list(range(362, 264, -1)) +
    list(range(0, 18)) +
    list(range(61, 292)) +
    [1, 10, 199]
)

def extract_from_any_format(data):
    """
    Essaie d'extraire left_hand, right_hand, face depuis n'importe quel format.
    """
    left = []
    right = []
    face = []

    # Cas 1 : dict
    if isinstance(data, dict):
        left = data.get("left_hand", [])
        right = data.get("right_hand", [])
        face = data.get("face", [])
        return left, right, face

    # Cas 2 : array d'objets
    if isinstance(data, np.ndarray) and data.dtype == object:
        for item in data:
            if isinstance(item, dict):
                left = item.get("left_hand", left)
                right = item.get("right_hand", right)
                face = item.get("face", face)
        return left, right, face

    # Cas 3 : array simple → probablement juste les landmarks
    if isinstance(data, np.ndarray):
        # Si 21 points → main seule
        if data.shape[0] == 21:
            return data, [], []
        # Si 468 points → visage seul
        if data.shape[0] == 468:
            return [], [], data

    return left, right, face


def safe_array(arr, expected_len):
    arr = np.array(arr)
    if arr.size == expected_len * 3:
        return arr.reshape(expected_len, 3)
    return np.zeros((expected_len, 3))


def choose_active_hand(left, right):
    if left.size > 0 and right.size == 0:
        return left, "LEFT", np.zeros((21,3))
    if right.size > 0 and left.size == 0:
        return right, "RIGHT", np.zeros((21,3))
    if left.size > 0 and right.size > 0:
        return (left, "LEFT", right) if left[:,2].mean() < right[:,2].mean() else (right, "RIGHT", left)
    return None, None, None


def extract_face(face):
    face = np.array(face)
    if face.shape[0] < 468:
        return np.zeros((len(FACE_INDICES), 3))
    return face[FACE_INDICES]


def restructure_dataset(input_dir="data", output_dir="clean_data"):
    os.makedirs(output_dir, exist_ok=True)

    for letter in string.ascii_uppercase:
        letter_dir = os.path.join(input_dir, letter)
        if not os.path.exists(letter_dir):
            continue

        print(f"\n=== Traitement de la lettre {letter} ===")

        out_letter_dir = os.path.join(output_dir, letter)
        os.makedirs(out_letter_dir, exist_ok=True)

        for filename in os.listdir(letter_dir):
            if not filename.endswith(".npy"):
                continue

            filepath = os.path.join(letter_dir, filename)
            data = np.load(filepath, allow_pickle=True)

            left, right, face_raw = extract_from_any_format(data)

            left = safe_array(left, 21)
            right = safe_array(right, 21)
            face = extract_face(face_raw)

            active, hand_type, passive = choose_active_hand(left, right)

            if active is None:
                print("⚠️ Aucune main détectée :", filename)
                continue

            combined = np.vstack([active, passive, face])

            new_name = filename.replace(".npy", f"_{hand_type}.npy")
            save_path = os.path.join(out_letter_dir, new_name)

            np.save(save_path, combined)
            print("✔ Sauvegardé :", new_name)

    print("\n🎉 Reconstruction terminée !")


if __name__ == "__main__":
    restructure_dataset()
