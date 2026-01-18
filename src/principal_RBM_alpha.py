"""
principal_RBM_alpha.py
Telecom SudParis - MAT5016

Script principal pour apprendre les caractères de la base Binary AlphaDigits
via un RBM et générer des caractères similaires.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results', 'RBM')
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)


def lire_alpha_digit(indices=None):
    """
    Récupère les données Binary AlphaDigits sous forme matricielle.
    
    Arguments:
        indices: liste des caractères à charger (0-9: chiffres, 10-35: lettres A-Z)
    
    Retourne:
        X: matrice (n_samples x n_pixels), une ligne = une donnée
    """
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, 'binaryalphadigs.mat'))
    data = mat['dat']
    
    X = []
    num_classes, num_samples = data.shape
    
    if indices is None:
        indices = range(num_classes)
        
    for i in indices:
        for j in range(num_samples):
            X.append(data[i, j].flatten())
            
    return np.array(X)


class RBM:
    """
    Structure RBM avec:
        W: matrice de poids (p x q)
        a: biais des unités visibles (p,)
        b: biais des unités cachées (q,)
    """
    def __init__(self, W, a, b):
        self.W = W
        self.a = a
        self.b = b


def sigmoid(x):
    """Fonction sigmoïde avec clipping pour stabilité."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def init_RBM(p, q):
    """
    Initialise un RBM.
    Poids: N(0, 0.01), biais: 0
    """
    W = np.random.normal(0, 0.01, (p, q))
    a = np.zeros(p)
    b = np.zeros(q)
    return RBM(W, a, b)


def entree_sortie_RBM(rbm, X):
    """Calcule P(h=1|v) via sigmoïde."""
    return sigmoid(X @ rbm.W + rbm.b)


def sortie_entree_RBM(rbm, H):
    """Calcule P(v=1|h) via sigmoïde."""
    return sigmoid(H @ rbm.W.T + rbm.a)


def train_RBM(rbm, X, epochs, lr, batch_size, k=1, verbose=True):
    """
    Apprentissage non supervisé par Contrastive-Divergence-k.
    
    Affiche l'erreur quadratique de reconstruction à chaque epoch.
    """
    n_samples, p = X.shape
    q = rbm.W.shape[1]
    errors = []
    
    iterator = tqdm(range(epochs), desc="Training RBM", disable=not verbose)
    for epoch in iterator:
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        
        total_error = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch = X_shuffled[i:min(i + batch_size, n_samples)]
            m = batch.shape[0]
            
            v0 = batch
            p_h0 = entree_sortie_RBM(rbm, v0)
            h = (np.random.rand(m, q) < p_h0).astype(float)
            
            for _ in range(k):
                p_v = sortie_entree_RBM(rbm, h)
                v = (np.random.rand(m, p) < p_v).astype(float)
                p_h = entree_sortie_RBM(rbm, v)
                h = (np.random.rand(m, q) < p_h).astype(float)
            
            vk, p_hk = v, p_h
            
            rbm.W += lr * (v0.T @ p_h0 - vk.T @ p_hk) / m
            rbm.a += lr * np.mean(v0 - vk, axis=0)
            rbm.b += lr * np.mean(p_h0 - p_hk, axis=0)
            
            total_error += np.mean((v0 - vk)**2)
            n_batches += 1
        
        avg_error = total_error / n_batches
        errors.append(avg_error)
        
        if verbose:
            iterator.set_postfix({"Reconstruction error": f"{avg_error:.4f}"})
    
    return rbm, errors


def generer_image_RBM(rbm, n_iter_gibbs, n_images, image_shape=(20, 16), show=True):
    """
    Génère des images via échantillonnage de Gibbs.
    """
    p = rbm.W.shape[0]
    q = rbm.W.shape[1]
    
    v = (np.random.rand(n_images, p) < 0.5).astype(float)
    
    for _ in range(n_iter_gibbs):
        h_prob = entree_sortie_RBM(rbm, v)
        h = (np.random.rand(n_images, q) < h_prob).astype(float)
        v_prob = sortie_entree_RBM(rbm, h)
        v = (np.random.rand(n_images, p) < v_prob).astype(float)
    
    if show:
        plt.figure(figsize=(2 * n_images, 2.5))
        plt.suptitle(f"Images générées par RBM (Gibbs iter={n_iter_gibbs})", fontsize=12)
        for i in range(n_images):
            plt.subplot(1, n_images, i + 1)
            plt.imshow(v[i].reshape(image_shape), cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return v


if __name__ == "__main__":
    
    print("="*70)
    print("Étude préliminaire - RBM sur Binary AlphaDigits")
    print("="*70)
    
    N_HIDDEN = 200
    EPOCHS = 100
    LR = 0.1
    BATCH_SIZE = 10
    N_GIBBS = 100
    
    print(f"\nParamètres: hidden={N_HIDDEN}, epochs={EPOCHS}, lr={LR}, batch={BATCH_SIZE}")
    
    alphabet_indices = list(range(10, 15))
    X = lire_alpha_digit(alphabet_indices)
    print(f"Données: {X.shape[0]} images, {X.shape[1]} pixels")
    
    plt.figure(figsize=(15, 4))
    random_indices = np.random.choice(X.shape[0], 5, replace=False)
    for i, idx in enumerate(random_indices):
        plt.subplot(1, 5, i+1)
        plt.imshow(X[idx].reshape(20, 16), cmap='gray')
        plt.axis('off')
    plt.suptitle("Exemples de données d'entrainement (Aléatoire)")
    plt.show()
    
    print("\n--- ENTRAINEMENT DU RBM ---")
    rbm = init_RBM(p=X.shape[1], q=N_HIDDEN)
    rbm, errors = train_RBM(rbm, X, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE)
    
    plt.figure(figsize=(10, 5))
    plt.plot(errors, 'b-', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Erreur de reconstruction")
    plt.title("Convergence du RBM")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n--- Génération d'images ---")
    generer_image_RBM(rbm, n_iter_gibbs=N_GIBBS, n_images=5)
    
    print("\n" + "="*70)
    print("Analyse 1: impact du nombre de neurones cachés")
    print("="*70)
    
    hidden_sizes = [100, 300, 700, 1000]
    fig, axes = plt.subplots(2, len(hidden_sizes), figsize=(4*len(hidden_sizes), 6))
    
    for idx, n_hidden in enumerate(hidden_sizes):
        print(f"Test q={n_hidden}...")
        rbm_test = init_RBM(p=X.shape[1], q=n_hidden)
        rbm_test, errs = train_RBM(rbm_test, X, epochs=50, lr=LR, batch_size=BATCH_SIZE, verbose=False)
        imgs = generer_image_RBM(rbm_test, N_GIBBS, 1, show=False)
        
        axes[0, idx].plot(errs)
        axes[0, idx].set_title(f"q={n_hidden}")
        axes[0, idx].set_xlabel("Epoch")
        if idx == 0:
            axes[0, idx].set_ylabel("Erreur (MSE)")
        axes[0, idx].grid(True, alpha=0.3)
        
        axes[1, idx].imshow(imgs[0].reshape(20, 16), cmap='gray')
        axes[1, idx].axis('off')
    
    plt.suptitle("Impact du nombre de neurones cachés", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "analyse_rbm_neurones.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("Analyse 2: pouvoir modélisant vs nombre de caractères")
    print("="*70)
    
    char_sets = [[10], [10, 11], [10, 11, 12], [10, 11, 12, 13], [10, 11, 12, 13, 14]]
    char_names = ["A", "A,B", "A,B,C", "A,B,C,D", "A-E"]
    
    fig, axes = plt.subplots(2, len(char_sets), figsize=(4*len(char_sets), 6))
    
    for idx, (chars, name) in enumerate(zip(char_sets, char_names)):
        print(f"Apprentissage sur: {name}")
        X_multi = lire_alpha_digit(chars)
        
        rbm_test = init_RBM(p=X_multi.shape[1], q=200)
        rbm_test, errs = train_RBM(rbm_test, X_multi, epochs=100, lr=LR, batch_size=BATCH_SIZE, verbose=False)
        imgs = generer_image_RBM(rbm_test, N_GIBBS, 1, show=False)
        
        axes[0, idx].plot(errs)
        axes[0, idx].set_title(f"Caractères: {name}")
        axes[0, idx].set_xlabel("Epoch")
        if idx == 0:
            axes[0, idx].set_ylabel("Erreur (MSE)")
        axes[0, idx].grid(True, alpha=0.3)
        
        axes[1, idx].imshow(imgs[0].reshape(20, 16), cmap='gray')
        axes[1, idx].axis('off')
    
    plt.suptitle("Pouvoir modélisant vs nombre de caractères", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "analyse_rbm_caracteres.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("Étude RBM terminée")
    print("="*70)
