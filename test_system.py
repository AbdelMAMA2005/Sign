#!/usr/bin/env python3
"""
Script de test pour vérifier le bon fonctionnement des modules
"""

import sys
import os

def test_imports():
    """Test l'importation des modules requis"""
    try:
        import cv2
        print("✓ OpenCV importé avec succès")
    except ImportError:
        print("✗ Erreur d'importation OpenCV")
        return False

    try:
        import mediapipe as mp
        print("✓ MediaPipe importé avec succès")
    except ImportError:
        print("✗ Erreur d'importation MediaPipe")
        return False

    try:
        import numpy as np
        print("✓ NumPy importé avec succès")
    except ImportError:
        print("✗ Erreur d'importation NumPy")
        return False

    try:
        from sklearn.ensemble import RandomForestClassifier
        print("✓ Scikit-learn importé avec succès")
    except ImportError:
        print("✗ Erreur d'importation Scikit-learn")
        return False

    try:
        import joblib
        print("✓ Joblib importé avec succès")
    except ImportError:
        print("✗ Erreur d'importation Joblib")
        return False

    return True

def test_hand_detector():
    """Test le module HandTrackingModule"""
    try:
        from HandTrackingModule import handDetector
        detector = handDetector()
        print("✓ HandTrackingModule importé et instancié avec succès")
        return True
    except Exception as e:
        print(f"✗ Erreur avec HandTrackingModule: {e}")
        return False

def test_data_directory():
    """Test l'existence du dossier data"""
    if os.path.exists('data'):
        print("✓ Dossier data existe")
        return True
    else:
        print("✗ Dossier data n'existe pas")
        return False

def main():
    print("=== Test du système de reconnaissance LSF ===\n")

    all_good = True

    print("1. Test des importations...")
    if not test_imports():
        all_good = False

    print("\n2. Test du détecteur de mains...")
    if not test_hand_detector():
        all_good = False

    print("\n3. Test du dossier de données...")
    if not test_data_directory():
        all_good = False

    print("\n" + "="*50)
    if all_good:
        print("✓ Tous les tests sont passés avec succès!")
        print("Vous pouvez maintenant utiliser le système.")
    else:
        print("✗ Certains tests ont échoué.")
        print("Vérifiez les erreurs ci-dessus et corrigez-les avant de continuer.")

    return all_good

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)