"""
compare_embeddings.py - Visualisation PCA des embeddings

Compare les embeddings du dataset avec les embeddings live
pour diagnostiquer les problèmes de cohérence.
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


def main():
    print("Chargement des embeddings du dataset...")
    
    try:
        emb_dataset = np.load("embeddings.npy")
        labels = np.load("labels.npy")
        print(f"Dataset: {emb_dataset.shape}")
    except:
        print("Erreur: embeddings.npy non trouve!")
        print("Executez d'abord generate_embeddings.py")
        return

    # Chercher les embeddings live
    live_files = [f for f in os.listdir('.') if f.startswith("live_embedding") and f.endswith(".npy")]
    
    if not live_files:
        print("\nAucun embedding live trouve.")
        print("Utilisez debug_live_embedding.py pour en capturer.")
        
        # Afficher juste le dataset
        print("\nVisualisation du dataset seul...")
        
        pca = PCA(n_components=2)
        proj = pca.fit_transform(emb_dataset)
        
        plt.figure(figsize=(12, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            pts = proj[mask]
            plt.scatter(pts[:, 0], pts[:, 1], 
                       c=[colors[i]], label=label, alpha=0.6, s=50)
        
        plt.legend(loc='best')
        plt.title("Embeddings du Dataset (PCA 2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("embeddings_pca.png", dpi=150)
        plt.show()
        return

    # Charger tous les embeddings live
    print(f"\nEmbeddings live trouves: {len(live_files)}")
    
    live_embeddings = []
    for f in sorted(live_files):
        emb = np.load(f)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        live_embeddings.append(emb[0])
        print(f"  {f}: shape {emb.shape}")
    
    live_embeddings = np.array(live_embeddings)
    print(f"Live total: {live_embeddings.shape}")

    # PCA sur tout
    print("\nCalcul PCA...")
    all_points = np.vstack([emb_dataset, live_embeddings])
    
    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_points)
    
    proj_dataset = proj[:len(emb_dataset)]
    proj_live = proj[len(emb_dataset):]

    # Visualisation
    plt.figure(figsize=(14, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # Dataset
    for i, label in enumerate(unique_labels):
        mask = labels == label
        pts = proj_dataset[mask]
        plt.scatter(pts[:, 0], pts[:, 1], 
                   c=[colors[i]], label=f"{label} (train)", 
                   alpha=0.5, s=40)
        
        # Centre du cluster
        center = pts.mean(axis=0)
        plt.annotate(label, center, fontsize=12, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Embeddings live
    for i, pt in enumerate(proj_live):
        plt.scatter(pt[0], pt[1], c='red', s=200, marker='X', 
                   edgecolors='black', linewidths=2,
                   label=f'LIVE #{i}' if i == 0 else '')
        plt.annotate(f'L{i}', (pt[0], pt[1]), fontsize=10,
                    xytext=(5, 5), textcoords='offset points')

    # Stats de distance
    print("\n=== ANALYSE DES DISTANCES ===")
    for i, live_emb in enumerate(live_embeddings):
        distances = np.linalg.norm(emb_dataset - live_emb, axis=1)
        
        # Top 3 plus proches
        closest_indices = np.argsort(distances)[:3]
        
        print(f"\nLive #{i}:")
        for j, idx in enumerate(closest_indices):
            print(f"  {j+1}. {labels[idx]}: distance = {distances[idx]:.4f}")
        
        print(f"  Distance moyenne: {distances.mean():.4f}")
        print(f"  Distance min: {distances.min():.4f}")

    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.title("Comparaison Embeddings: Dataset vs LIVE (PCA 2D)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("embeddings_comparison.png", dpi=150)
    print("\nGraphique sauvegarde: embeddings_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
