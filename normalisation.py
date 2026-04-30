import numpy as np

# ---------------------------------------------------------
# 🖐️ NORMALISATION DES MAINS - VERSION CORRIGÉE
# ---------------------------------------------------------
def normalize_hand(hand):
    """
    Normalise une main MediaPipe (21x3) pour être invariante
    à la distance et à la position dans le frame.
    
    CORRECTION: Suppression de la rotation qui causait des 
    incohérences entre l'entraînement et la prédiction live.
    
    Étapes:
    1) Centre sur le poignet (landmark 0)
    2) Normalise par la distance poignet → majeur MCP
    3) PAS de rotation (source d'incohérence)
    
    Args:
        hand: np.array (21, 3) - landmarks bruts de la main
        
    Returns:
        np.array (21, 3) - landmarks normalisés
    """
    # Vérification de la forme
    if hand.shape != (21, 3):
        return np.zeros((21, 3))
    
    # Vérification que la main contient des données
    if np.abs(hand).sum() < 1e-6:
        return np.zeros((21, 3))

    hand = hand.copy().astype(np.float32)

    # 1) CENTRAGE sur le poignet (landmark 0)
    wrist = hand[0].copy()
    hand_centered = hand - wrist

    # 2) NORMALISATION par distance poignet → majeur MCP
    # Le majeur MCP (landmark 9) est un point stable
    middle_mcp = hand_centered[9]
    
    # Distance euclidienne 3D (plus robuste que 2D)
    ref_dist = np.linalg.norm(middle_mcp)
    
    if ref_dist < 1e-6:
        # Fallback: utiliser la distance max
        ref_dist = np.max(np.linalg.norm(hand_centered, axis=1))
        if ref_dist < 1e-6:
            return np.zeros((21, 3))

    hand_normalized = hand_centered / ref_dist
    
    # NOTE: Pas de rotation!
    # La rotation basée sur arctan2 peut donner des résultats
    # différents selon l'orientation de la caméra/miroir,
    # causant des incohérences entre train et live.
    
    return hand_normalized


def compute_hand_features(hand_normalized):
    """
    Calcule des features additionnelles invariantes.
    
    OPTIONNEL: Pour un modèle plus robuste, on peut ajouter
    des distances inter-landmarks.
    """
    if hand_normalized.shape != (21, 3):
        return hand_normalized.flatten()
    
    # Pour l'instant, on retourne juste les coordonnées
    # Les distances peuvent être ajoutées plus tard si besoin
    return hand_normalized.flatten()
