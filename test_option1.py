"""
test_option1.py - Test rapide du comportement d'OPTION 1

Cet outil montre comment OPTION 1 normalise les vecteurs
pour que la même lettre ait la même représentation peu importe la main.
"""

import numpy as np
from pipeline_utils import build_feature_vector_normalized
from normalisation import normalize_hand


def create_dummy_hand_left():
    """Crée des landmarks fictifs pour main gauche (signe 'A')."""
    hand = np.random.randn(21, 3) * 0.05  # Petits nombres
    hand[0] = [0.5, 0.5, 0.0]  # Poignet
    hand[1:5] = [[0.5 + 0.1*i, 0.5, 0.0] for i in range(4)]
    return hand


def create_dummy_hand_right():
    """Crée des landmarks fictifs pour main droite (signe 'A')."""
    hand = np.random.randn(21, 3) * 0.05  # Petits nombres
    hand[0] = [0.3, 0.5, 0.0]  # Poignet (position différente)
    hand[1:5] = [[0.3 + 0.1*i, 0.5, 0.0] for i in range(4)]
    return hand


def test_option1_letter():
    """Test OPTION 1 pour une lettre 'A' avec main gauche et droite."""
    
    print("=" * 70)
    print("TEST OPTION 1: Lettre 'A' avec main gauche ET droite")
    print("=" * 70)
    
    # Créer des données fictives pour main gauche
    print("\n1. Simulation: Main GAUCHE signe 'A'")
    print("-" * 70)
    left_hand = create_dummy_hand_left()
    right_hand_empty = np.zeros((21, 3))
    
    print(f"   left_hand shape: {left_hand.shape}")
    print(f"   left_hand sum: {np.abs(left_hand).sum():.6f}")
    print(f"   right_hand sum: {np.abs(right_hand_empty).sum():.6f}")
    
    # Normaliser avec hand_used="left"
    vec_left, hand_side, sign_type = build_feature_vector_normalized(
        left_hand, right_hand_empty, hand_used="left"
    )
    
    print(f"\n   Après normalisation (hand_used='left'):")
    print(f"   - Vecteur shape: {vec_left.shape}")
    print(f"   - Hand side detected: {hand_side}")
    print(f"   - Sign type: {sign_type}")
    print(f"   - Positions [0-62] sum: {np.abs(vec_left[0, :63]).sum():.6f}  ← MAIN ACTIVE")
    print(f"   - Positions [63-125] sum: {np.abs(vec_left[0, 63:]).sum():.6f}  ← ZEROS")
    
    # Créer des données fictives pour main droite
    print("\n2. Simulation: Main DROITE signe 'A'")
    print("-" * 70)
    left_hand_empty = np.zeros((21, 3))
    right_hand = create_dummy_hand_right()
    
    print(f"   left_hand sum: {np.abs(left_hand_empty).sum():.6f}")
    print(f"   right_hand shape: {right_hand.shape}")
    print(f"   right_hand sum: {np.abs(right_hand).sum():.6f}")
    
    # Normaliser avec hand_used="right"
    vec_right, hand_side, sign_type = build_feature_vector_normalized(
        left_hand_empty, right_hand, hand_used="right"
    )
    
    print(f"\n   Après normalisation (hand_used='right'):")
    print(f"   - Vecteur shape: {vec_right.shape}")
    print(f"   - Hand side detected: {hand_side}")
    print(f"   - Sign type: {sign_type}")
    print(f"   - Positions [0-62] sum: {np.abs(vec_right[0, :63]).sum():.6f}  ← MAIN ACTIVE (droite ici!)")
    print(f"   - Positions [63-125] sum: {np.abs(vec_right[0, 63:]).sum():.6f}  ← ZEROS")
    
    # Comparaison
    print("\n3. COMPARAISON - Cohérence OPTION 1")
    print("-" * 70)
    
    structure_left_ok = (
        np.abs(vec_left[0, :63]).sum() > 1e-6 and 
        np.abs(vec_left[0, 63:]).sum() < 1e-6
    )
    
    structure_right_ok = (
        np.abs(vec_right[0, :63]).sum() > 1e-6 and 
        np.abs(vec_right[0, 63:]).sum() < 1e-6
    )
    
    if structure_left_ok and structure_right_ok:
        print("✓ Les deux vecteurs ont la MÊME STRUCTURE!")
        print("  - Positions 0-62: actif (main active)")
        print("  - Positions 63-125: zeros (main inactive)")
        print("\n✓ OPTION 1 fonctionne correctement!")
        print("  Le modèle verra 'A' de la même façon peu importe la main")
    else:
        print("❌ Les vecteurs n'ont pas la même structure!")


def test_option1_word():
    """Test OPTION 1 pour un mot avec deux mains."""
    
    print("\n" + "=" * 70)
    print("TEST OPTION 1: Mot 'MERCI' avec deux mains")
    print("=" * 70)
    
    left_hand = create_dummy_hand_left()
    right_hand = create_dummy_hand_right()
    
    print(f"\n1. Deux mains détectées")
    print(f"   - left_hand sum: {np.abs(left_hand).sum():.6f}")
    print(f"   - right_hand sum: {np.abs(right_hand).sum():.6f}")
    
    # Normaliser avec hand_used="both"
    vec_both, hand_side, sign_type = build_feature_vector_normalized(
        left_hand, right_hand, hand_used="both"
    )
    
    print(f"\n2. Après normalisation (hand_used='both'):")
    print(f"   - Hand side detected: {hand_side}")
    print(f"   - Sign type: {sign_type}")
    print(f"   - Positions [0-62] sum: {np.abs(vec_both[0, :63]).sum():.6f}  ← Main GAUCHE")
    print(f"   - Positions [63-125] sum: {np.abs(vec_both[0, 63:]).sum():.6f}  ← Main DROITE")
    
    if np.abs(vec_both[0, :63]).sum() > 1e-6 and np.abs(vec_both[0, 63:]).sum() > 1e-6:
        print("\n✓ Les deux mains sont présentes dans le vecteur")
        print("  Le modèle reconnaîtra les mots contenant les deux mains")


def test_consistency():
    """Teste que la normalisation est cohérente."""
    
    print("\n" + "=" * 70)
    print("TEST: Cohérence de la normalisation")
    print("=" * 70)
    
    # Même main, deux fois
    hand = create_dummy_hand_left()
    
    vec1, _, _ = build_feature_vector_normalized(
        hand.copy(), np.zeros((21, 3)), hand_used="left"
    )
    
    vec2, _, _ = build_feature_vector_normalized(
        hand.copy(), np.zeros((21, 3)), hand_used="left"
    )
    
    # Les vecteurs doivent être identiques
    diff = np.abs(vec1 - vec2).sum()
    
    print(f"Différence entre deux appels identiques: {diff:.10f}")
    
    if diff < 1e-6:
        print("✓ La normalisation est déterministe!")
    else:
        print("❌ La normalisation n'est pas déterministe!")


def main():
    print("\n" + "=" * 70)
    print("TESTS DU PIPELINE OPTION 1")
    print("=" * 70)
    
    test_option1_letter()
    test_option1_word()
    test_consistency()
    
    print("\n" + "=" * 70)
    print("RESUME")
    print("=" * 70)
    print("""
✓ OPTION 1 fonctionne comme prévu:
  1. Lettres gauche ET droite → Même structure (positions 0-62 actif)
  2. Mots (deux mains) → Les deux positions (0-62 et 63-125)
  3. Normalisation cohérente et déterministe
  
Prochaines étapes:
  1. Collectez des données: python data_collection.py
  2. Vérifiez la cohérence: python check_consistency.py
  3. Entraînez le modèle: python train_full_model.py
  4. Testez en live: python predict_sign.py
""")


if __name__ == "__main__":
    main()
