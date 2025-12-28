"""
principal_DBN_alpha.py
Telecom SudParis - MAT5016

Script principal pour apprendre les caractères de la base Binary AlphaDigits
via un DBN (Deep Belief Network) et générer des caractères similaires.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================================
# FONCTIONS DE CHARGEMENT DES DONNÉES
# ============================================================================

def lire_alpha_digit(indices=None):
    """
    Récupère les données Binary AlphaDigits sous forme matricielle.
    
    Arguments:
        indices: liste des caractères à charger (0-9: chiffres, 10-35: lettres A-Z)
    
    Retourne:
        X: matrice (n_samples x n_pixels), une ligne = une donnée
    """
    mat = scipy.io.loadmat('binaryalphadigs.mat')
    data = mat['dat']
    
    X = []
    num_classes, num_samples = data.shape
    
    if indices is None:
        indices = range(num_classes)
        
    for i in indices:
        for j in range(num_samples):
            X.append(data[i, j].flatten())
            
    return np.array(X)

# ============================================================================
# STRUCTURE RBM
# ============================================================================

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

# ============================================================================
# FONCTIONS RBM
# ============================================================================

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
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        
        total_error = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch = X_shuffled[i:min(i + batch_size, n_samples)]
            m = batch.shape[0]
            
            # Phase positive
            v0 = batch
            p_h0 = entree_sortie_RBM(rbm, v0)
            h = (np.random.rand(m, q) < p_h0).astype(float)
            
            # CD-k
            for _ in range(k):
                p_v = sortie_entree_RBM(rbm, h)
                v = (np.random.rand(m, p) < p_v).astype(float)
                p_h = entree_sortie_RBM(rbm, v)
                h = (np.random.rand(m, q) < p_h).astype(float)
            
            vk, p_hk = v, p_h
            
            # Gradients et mise à jour
            rbm.W += lr * (v0.T @ p_h0 - vk.T @ p_hk) / m
            rbm.a += lr * np.mean(v0 - vk, axis=0)
            rbm.b += lr * np.mean(p_h0 - p_hk, axis=0)
            
            total_error += np.mean((v0 - vk)**2)
            n_batches += 1
        
        avg_error = total_error / n_batches
        errors.append(avg_error)
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs} - Erreur reconstruction: {avg_error:.4f}")
    
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

# ============================================================================
# STRUCTURE DBN
# ============================================================================

class DBN:
    """
    Structure DBN: liste de RBMs empilés.
    """
    def __init__(self, rbms):
        self.rbms = rbms  # Liste de RBMs

# ============================================================================
# FONCTIONS DBN
# ============================================================================

def init_DBN(layer_sizes):
    """
    Initialise un DBN avec des couches de tailles données.
    
    Arguments:
        layer_sizes: [p, q1, q2, ...] où p = dim visible, qi = dim cachée couche i
    
    Retourne:
        DBN initialisé
    """
    rbms = []
    for i in range(len(layer_sizes) - 1):
        rbm = init_RBM(layer_sizes[i], layer_sizes[i+1])
        rbms.append(rbm)
    return DBN(rbms)


def train_DBN(dbn, X, epochs, lr, batch_size, k=1, verbose=True):
    """
    Apprentissage "greedy layer-wise":
    - Entraine chaque RBM l'un après l'autre
    - L'entrée du RBM i+1 = sortie cachée (probabilités) du RBM i
    """
    n_rbms = len(dbn.rbms)
    current_input = X.copy()
    all_errors = []
    
    for i, rbm in enumerate(dbn.rbms):
        if verbose:
            print(f"\n--- Entrainement RBM couche {i+1}/{n_rbms} ---")
            print(f"    Dimensions: {rbm.W.shape[0]} -> {rbm.W.shape[1]}")
        
        rbm, errors = train_RBM(rbm, current_input, epochs, lr, batch_size, k, verbose)
        all_errors.append(errors)
        
        # Propagation pour la couche suivante
        current_input = entree_sortie_RBM(rbm, current_input)
    
    return dbn, all_errors


def generer_image_DBN(dbn, n_iter_gibbs, n_images, image_shape=(20, 16), show=True):
    """
    Génère des images à partir du DBN:
    1. Échantillonnage de Gibbs sur le RBM du sommet
    2. Propagation inverse couche par couche
    """
    # RBM du sommet
    top_rbm = dbn.rbms[-1]
    q_top = top_rbm.W.shape[1]
    
    # Initialisation aléatoire des unités cachées du sommet
    h = (np.random.rand(n_images, q_top) < 0.5).astype(float)
    
    # Gibbs sampling sur le RBM du sommet
    for _ in range(n_iter_gibbs):
        v_prob = sortie_entree_RBM(top_rbm, h)
        v = (np.random.rand(n_images, top_rbm.W.shape[0]) < v_prob).astype(float)
        h_prob = entree_sortie_RBM(top_rbm, v)
        h = (np.random.rand(n_images, q_top) < h_prob).astype(float)
    
    # Propagation inverse
    current = v
    for rbm in reversed(dbn.rbms[:-1]):
        current_prob = sortie_entree_RBM(rbm, current)
        current = (np.random.rand(n_images, rbm.W.shape[0]) < current_prob).astype(float)
    
    if show:
        plt.figure(figsize=(2 * n_images, 2.5))
        plt.suptitle(f"Images générées par DBN ({len(dbn.rbms)} couches, Gibbs={n_iter_gibbs})", fontsize=12)
        for i in range(n_images):
            plt.subplot(1, n_images, i + 1)
            plt.imshow(current[i].reshape(image_shape), cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return current

# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("ÉTUDE DBN - DEEP BELIEF NETWORK SUR BINARY ALPHADIGITS")
    print("="*70)
    
    # Paramètres
    EPOCHS = 100
    LR = 0.1
    BATCH_SIZE = 10
    N_GIBBS = 200
    
    print(f"\nParamètres: epochs={EPOCHS}, lr={LR}, batch={BATCH_SIZE}")
    
    # Chargement données
    caracteres = [10]  # 'A'
    X = lire_alpha_digit(caracteres)
    print(f"Données: {X.shape[0]} images, {X.shape[1]} pixels (20x16)")
    
    # Affichage exemples
    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(X[i].reshape(20, 16), cmap='gray')
        plt.axis('off')
    plt.suptitle("Exemples de données d'entrainement")
    plt.show()
    
    # =========================================
    # TEST DBN SIMPLE (2 couches)
    # =========================================
    print("\n" + "="*70)
    print("TEST DBN 2 COUCHES: 320 -> 200 -> 100")
    print("="*70)
    
    layer_sizes = [320, 200, 100]
    dbn = init_DBN(layer_sizes)
    dbn, all_errors = train_DBN(dbn, X, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE)
    
    # Courbes d'erreur
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, errors in enumerate(all_errors):
        axes[i].plot(errors, 'b-', linewidth=2)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Erreur reconstruction")
        axes[i].set_title(f"Couche {i+1}")
        axes[i].grid(True, alpha=0.3)
    plt.suptitle("Convergence DBN couche par couche", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Génération
    print("\n--- GÉNÉRATION D'IMAGES ---")
    generer_image_DBN(dbn, n_iter_gibbs=N_GIBBS, n_images=5)
    
    # =========================================
    # ANALYSE 1: IMPACT DE LA PROFONDEUR
    # =========================================
    print("\n" + "="*70)
    print("ANALYSE 1: IMPACT DE LA PROFONDEUR DU DBN")
    print("="*70)
    
    architectures = [
        [320, 200],           # 1 couche
        [320, 200, 100],      # 2 couches
        [320, 200, 100, 50],  # 3 couches
    ]
    arch_names = ["1 couche", "2 couches", "3 couches"]
    
    fig, axes = plt.subplots(1, len(architectures), figsize=(5*len(architectures), 4))
    
    for idx, (arch, name) in enumerate(zip(architectures, arch_names)):
        print(f"\nTest: {name} - {arch}")
        dbn_test = init_DBN(arch)
        dbn_test, _ = train_DBN(dbn_test, X, epochs=50, lr=LR, batch_size=BATCH_SIZE, verbose=False)
        imgs = generer_image_DBN(dbn_test, N_GIBBS, 1, show=False)
        
        axes[idx].imshow(imgs[0].reshape(20, 16), cmap='gray')
        axes[idx].set_title(f"{name}\n{arch}")
        axes[idx].axis('off')
    
    plt.suptitle("Impact de la profondeur du DBN", fontsize=14)
    plt.tight_layout()
    plt.savefig("analyse_dbn_profondeur.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # =========================================
    # ANALYSE 2: PLUSIEURS CARACTÈRES
    # =========================================
    print("\n" + "="*70)
    print("ANALYSE 2: DBN SUR PLUSIEURS CARACTÈRES")
    print("="*70)
    
    char_sets = [[10], [10, 11, 12], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
    char_names = ["A seul", "A-C", "A-J (10 lettres)"]
    
    fig, axes = plt.subplots(len(char_sets), 5, figsize=(12, 3*len(char_sets)))
    
    for row, (chars, name) in enumerate(zip(char_sets, char_names)):
        print(f"\nApprentissage sur: {name}")
        X_multi = lire_alpha_digit(chars)
        print(f"  {X_multi.shape[0]} images")
        
        dbn_test = init_DBN([320, 200, 100])
        dbn_test, _ = train_DBN(dbn_test, X_multi, epochs=100, lr=LR, batch_size=BATCH_SIZE, verbose=False)
        imgs = generer_image_DBN(dbn_test, N_GIBBS, 5, show=False)
        
        for col in range(5):
            axes[row, col].imshow(imgs[col].reshape(20, 16), cmap='gray')
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(name, fontsize=10, rotation=0, labelpad=40, ha='right')
    
    plt.suptitle("Génération DBN selon le nombre de caractères appris", fontsize=14)
    plt.tight_layout()
    plt.savefig("analyse_dbn_caracteres.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # =========================================
    # COMPARAISON RBM vs DBN
    # =========================================
    print("\n" + "="*70)
    print("COMPARAISON RBM (1 couche) vs DBN (2 couches)")
    print("="*70)
    
    X_multi = lire_alpha_digit([10, 11, 12, 13, 14])
    
    print("\nEntrainement RBM simple (320 -> 200)...")
    rbm_simple = init_RBM(320, 200)
    rbm_simple, _ = train_RBM(rbm_simple, X_multi, epochs=100, lr=LR, batch_size=BATCH_SIZE, verbose=False)
    rbm_imgs = generer_image_RBM(rbm_simple, N_GIBBS, 5, show=False)
    
    print("Entrainement DBN (320 -> 200 -> 100)...")
    dbn_compare = init_DBN([320, 200, 100])
    dbn_compare, _ = train_DBN(dbn_compare, X_multi, epochs=100, lr=LR, batch_size=BATCH_SIZE, verbose=False)
    dbn_imgs = generer_image_DBN(dbn_compare, N_GIBBS, 5, show=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(5):
        axes[0, i].imshow(rbm_imgs[i].reshape(20, 16), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(dbn_imgs[i].reshape(20, 16), cmap='gray')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel("RBM", fontsize=12, rotation=0, labelpad=30, ha='right')
    axes[1, 0].set_ylabel("DBN", fontsize=12, rotation=0, labelpad=30, ha='right')
    
    plt.suptitle("Comparaison RBM vs DBN sur caractères A-E", fontsize=14)
    plt.tight_layout()
    plt.savefig("comparaison_rbm_dbn.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("ÉTUDE DBN TERMINÉE")
    print("="*70)
