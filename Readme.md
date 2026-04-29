content = """# ✋🤟 Reconnaissance de Signes - Langue des Signes Française (LSF)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-green.svg)](https://google.github.io/mediapipe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Sign** est un système de reconnaissance de la LSF en temps réel conçu pour briser les barrières sociales. Ce projet utilise la vision par ordinateur pour **collecter**, **entraîner** et **prédire** des signes à partir des landmarks MediaPipe (mains et visage).

---

## Fonctionnalités

* **Collecte Dynamique** : Détection intelligente des mains (gauche/droite/deux mains) et extraction de **90 points faciaux** pour capturer l'expression des signes-mots.
* **Pipeline de Données** : Structuration automatique des fichiers `.npy` pour un entraînement propre.
* **Architecture MLP** : Modèle TensorFlow Dense avec un vecteur d'entrée de **396 features**.
* **Interface de Prédiction** : Affichage en temps réel avec stabilisation anti-clignotement et retour visuel coloré.

---

## Structure du projet

```text
.
├── TrackingModule.py   # Wrapper MediaPipe Holistic (Cœur du suivi)
├── data_collection.py  # Script de création du dataset
├── train_model.py      # Script d'entraînement (MLP)
├── predict_sign.py     # Script de démonstration en temps réel
├── requirements.txt    # Liste des dépendances
├── data/               # Dataset local (ignorer dans Git)
└── models/             # Modèles exportés (.h5) et encodeurs