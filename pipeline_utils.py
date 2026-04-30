import numpy as np
from normalisation import normalize_hand

# ---------------------------------------------------------
# 🔧 CONFIGURATION - SANS VISAGE
# ---------------------------------------------------------
LEFT_SIZE = 21 * 3   # 63 features pour main gauche
RIGHT_SIZE = 21 * 3  # 63 features pour main droite
INPUT_SIZE = LEFT_SIZE + RIGHT_SIZE  # 126 total (PAS de visage)


def detect_hands(left_raw, right_raw):
    """
    Détecte si les mains sont présentes.
    
    CORRECTION: Vérification plus robuste avec seuil de somme.
    
    Args:
        left_raw: np.array (21, 3) - landmarks main gauche bruts
        right_raw: np.array (21, 3) - landmarks main droite bruts
        
    Returns:
        tuple (bool, bool): (left_detected, right_detected)
    """
    left_detected = (
        left_raw.shape == (21, 3) and 
        np.abs(left_raw).sum() > 1e-6
    )
    right_detected = (
        right_raw.shape == (21, 3) and 
        np.abs(right_raw).sum() > 1e-6
    )
    return left_detected, right_detected


def detect_hand_side(left_detected, right_detected):
    """
    Retourne l'information sur quelle(s) main(s) et le type.
    
    Args:
        left_detected: bool - main gauche détectée
        right_detected: bool - main droite détectée
        
    Returns:
        tuple (str, str): (hand_side, sign_type)
        - ('left', 'letter') - main gauche seule = lettre
        - ('right', 'letter') - main droite seule = lettre
        - ('both', 'word') - deux mains = mot
        - (None, None) - aucune main
    """
    if left_detected and not right_detected:
        return 'left', 'letter'
    elif right_detected and not left_detected:
        return 'right', 'letter'
    elif left_detected and right_detected:
        return 'both', 'word'
    else:
        return None, None


def build_feature_vector(left_raw, right_raw, face_raw=None):
    """
    Construit le vecteur de features pour le modèle.
    
    VERSION STANDARD: Garde left aux positions 0-63 et right aux 63-126
    même si c'est la main droite qui signe (pour cohérence avec mots).
    
    IMPORTANT: Cette fonction DOIT être utilisée PARTOUT:
    - data_collection.py (pas nécessaire, on sauve le brut)
    - build_dataset.py (normalise depuis le brut)
    - train_full_model.py (normalise depuis le brut)
    - predict_sign.py (normalise le live)
    - debug_live_embedding.py (normalise le live)
    
    Le visage est complètement ignoré pour éviter le biais.
    
    Args:
        left_raw: np.array (21, 3) - landmarks main gauche BRUTS
        right_raw: np.array (21, 3) - landmarks main droite BRUTS
        face_raw: ignoré (gardé pour compatibilité)
        
    Returns:
        tuple: (vec, hand_side, sign_type)
        - vec: np.array (1, 126) ou None si aucune main
        - hand_side: 'left', 'right', 'both', ou None
        - sign_type: 'letter', 'word', ou None
    """
    # Détection des mains
    left_detected, right_detected = detect_hands(left_raw, right_raw)

    # Aucune main → pas de prédiction
    if not left_detected and not right_detected:
        return None, None, None

    # Normalisation des mains
    # IMPORTANT: normalize_hand retourne zeros si la main n'est pas détectée
    left_norm = normalize_hand(left_raw).flatten()
    right_norm = normalize_hand(right_raw).flatten()

    # Construction du vecteur final [left(63) + right(63)] = 126
    vec = np.zeros(INPUT_SIZE, dtype=np.float32)
    vec[:LEFT_SIZE] = left_norm
    vec[LEFT_SIZE:INPUT_SIZE] = right_norm

    # Info sur la main et le type
    hand_side, sign_type = detect_hand_side(left_detected, right_detected)
    
    return vec.reshape(1, -1), hand_side, sign_type


def build_feature_vector_normalized(left_raw, right_raw, hand_used=None, face_raw=None):
    """
    VERSION OPTION 1: Main active toujours aux positions 0-63.
    
    Utile si vous collectez avec _left ou _right dans les noms de fichiers
    et voulez normaliser la position quelle que soit la main.
    
    Args:
        left_raw: np.array (21, 3) - landmarks main gauche BRUTS
        right_raw: np.array (21, 3) - landmarks main droite BRUTS
        hand_used: str - "left", "right", ou "both" (optionnel, détecté auto si None)
        face_raw: ignoré (gardé pour compatibilité)
        
    Returns:
        tuple: (vec, hand_side, sign_type)
        - vec: np.array (1, 126) - main active en [0-63], zeros en [63-126] pour lettres
        - hand_side: 'left', 'right', 'both', ou None
        - sign_type: 'letter', 'word', ou None
    """
    # Détection des mains
    left_detected, right_detected = detect_hands(left_raw, right_raw)

    # Aucune main → pas de prédiction
    if not left_detected and not right_detected:
        return None, None, None

    # Si hand_used n'est pas fourni, le détecter
    if hand_used is None:
        hand_side, sign_type = detect_hand_side(left_detected, right_detected)
        hand_used = hand_side
    else:
        hand_side, sign_type = detect_hand_side(left_detected, right_detected)

    # Normalisation des mains détectées
    left_norm = normalize_hand(left_raw).flatten() if left_detected else np.zeros(LEFT_SIZE)
    right_norm = normalize_hand(right_raw).flatten() if right_detected else np.zeros(RIGHT_SIZE)

    # Construction du vecteur selon le hand_used
    vec = np.zeros(INPUT_SIZE, dtype=np.float32)
    
    if hand_used == "both":
        # Mots : toujours [left + right]
        vec[:LEFT_SIZE] = left_norm
        vec[LEFT_SIZE:INPUT_SIZE] = right_norm
    elif hand_used == "left":
        # Lettre gauche : main active en [0-63], zeros en [63-126]
        vec[:LEFT_SIZE] = left_norm
        vec[LEFT_SIZE:INPUT_SIZE] = np.zeros(RIGHT_SIZE)
    elif hand_used == "right":
        # Lettre droite : main active en [0-63], zeros en [63-126]
        # OPTION 1: Mettre la droite aux positions gauche
        vec[:LEFT_SIZE] = right_norm
        vec[LEFT_SIZE:INPUT_SIZE] = np.zeros(RIGHT_SIZE)
    else:
        # Fallback: standard
        vec[:LEFT_SIZE] = left_norm
        vec[LEFT_SIZE:INPUT_SIZE] = right_norm
    
    return vec.reshape(1, -1), hand_side, sign_type


def debug_feature_vector(left_raw, right_raw):
    """
    Version debug qui affiche les stats du vecteur.
    Utile pour comparer train vs live.
    """
    vec, hand_side, sign_type = build_feature_vector(left_raw, right_raw)
    
    if vec is None:
        print("[DEBUG] Aucune main détectée")
        return None
    
    print(f"[DEBUG] Main: {hand_side}, Type: {sign_type}")
    print(f"[DEBUG] Vec shape: {vec.shape}")
    print(f"[DEBUG] Vec mean: {vec.mean():.6f}")
    print(f"[DEBUG] Vec std: {vec.std():.6f}")
    print(f"[DEBUG] Vec min: {vec.min():.6f}, max: {vec.max():.6f}")
    print(f"[DEBUG] Left part sum: {np.abs(vec[0, :LEFT_SIZE]).sum():.6f}")
    print(f"[DEBUG] Right part sum: {np.abs(vec[0, LEFT_SIZE:]).sum():.6f}")
    
    return vec, hand_side, sign_type
