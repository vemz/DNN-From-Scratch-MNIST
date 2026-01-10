"""
principal_DNN_MNIST.py
Telecom SudParis - MAT5016

Script principal pour classifier MNIST avec un DNN.
Compare pré-entrainement DBN vs initialisation aléatoire.
Génère 3 figures d'analyse: profondeur, largeur, taille des données.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results', 'DNN')
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================================
# FONCTIONS DE CHARGEMENT DES DONNÉES
# ============================================================================

def load_mnist_binarized(n_samples=None, threshold=0.5):
    """
    Charge MNIST et binarise les images.
    
    Retourne:
        X_train, X_test, y_train, y_test
    """
    print("Chargement MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(float) / 255.0
    y = mnist.target.astype(int)
    
    # Binarisation
    X = (X > threshold).astype(float)
    
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"MNIST: {X_train.shape[0]} train, {X_test.shape[0]} test, {X.shape[1]} pixels")
    return X_train, X_test, y_train, y_test


def one_hot(y, n_classes=10):
    """Encode les labels en one-hot."""
    n = len(y)
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1
    return Y

# ============================================================================
# STRUCTURE RBM
# ============================================================================

class RBM:
    """Structure RBM avec W, a, b."""
    def __init__(self, W, a, b):
        self.W = W
        self.a = a
        self.b = b

# ============================================================================
# FONCTIONS RBM
# ============================================================================

def sigmoid(x):
    """Sigmoïde avec clipping."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def init_RBM(p, q):
    """Initialise un RBM."""
    W = np.random.normal(0, 0.01, (p, q))
    a = np.zeros(p)
    b = np.zeros(q)
    return RBM(W, a, b)


def entree_sortie_RBM(rbm, X):
    """P(h=1|v)"""
    return sigmoid(X @ rbm.W + rbm.b)


def sortie_entree_RBM(rbm, H):
    """P(v=1|h)"""
    return sigmoid(H @ rbm.W.T + rbm.a)


def train_RBM(rbm, X, epochs, lr, batch_size, k=1, verbose=False):
    """Apprentissage CD-k."""
    n_samples, p = X.shape
    q = rbm.W.shape[1]
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        
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
    
    return rbm

# ============================================================================
# STRUCTURE DBN
# ============================================================================

class DBN:
    """Structure DBN: liste de RBMs."""
    def __init__(self, rbms):
        self.rbms = rbms

# ============================================================================
# FONCTIONS DBN
# ============================================================================

def init_DBN(layer_sizes):
    """Initialise un DBN."""
    rbms = []
    for i in range(len(layer_sizes) - 1):
        rbm = init_RBM(layer_sizes[i], layer_sizes[i+1])
        rbms.append(rbm)
    return DBN(rbms)


def train_DBN(dbn, X, epochs, lr, batch_size, k=1, verbose=False):
    """Apprentissage greedy layer-wise."""
    current_input = X.copy()
    
    for i, rbm in enumerate(dbn.rbms):
        if verbose:
            print(f"  DBN: couche {i+1}/{len(dbn.rbms)}")
        rbm = train_RBM(rbm, current_input, epochs, lr, batch_size, k, verbose)
        current_input = entree_sortie_RBM(rbm, current_input)
    
    return dbn

# ============================================================================
# STRUCTURE DNN
# ============================================================================

class DNN:
    """
    Structure DNN pour classification:
        dbn: DBN pré-entrainé
        W_class: poids couche classification
        b_class: biais couche classification
    """
    def __init__(self, dbn, W_class, b_class):
        self.dbn = dbn
        self.W_class = W_class
        self.b_class = b_class

# ============================================================================
# FONCTIONS DNN
# ============================================================================

def init_DNN(layer_sizes, n_classes=10):
    """
    Initialise un DNN avec initialisation aléatoire.
    
    Arguments:
        layer_sizes: [p, q1, q2, ...] architecture du DBN
        n_classes: nombre de classes
    """
    dbn = init_DBN(layer_sizes)
    last_hidden = layer_sizes[-1]
    W_class = np.random.normal(0, 0.01, (last_hidden, n_classes))
    b_class = np.zeros(n_classes)
    return DNN(dbn, W_class, b_class)


def pretrain_DNN(dnn, X, epochs, lr, batch_size, k=1, verbose=False):
    """
    Pré-entraine le DNN via DBN (non supervisé).
    """
    if verbose:
        print("Pré-entrainement DBN...")
    dnn.dbn = train_DBN(dnn.dbn, X, epochs, lr, batch_size, k, verbose)
    return dnn


def calcul_softmax(z):
    """Softmax stable numériquement."""
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def entree_sortie_reseau(dnn, X):
    """
    Propagation avant complète.
    
    Retourne:
        activations: liste des activations de chaque couche
        output: sortie softmax finale
    """
    activations = [X]
    current = X
    
    # Propagation dans le DBN
    for rbm in dnn.dbn.rbms:
        current = sigmoid(current @ rbm.W + rbm.b)
        activations.append(current)
    
    # Couche de classification
    logits = current @ dnn.W_class + dnn.b_class
    output = calcul_softmax(logits)
    
    return activations, output


def retropropagation(dnn, X, Y, epochs, lr, batch_size, verbose=False):
    """
    Fine-tuning supervisé par rétropropagation.
    
    Arguments:
        X: données d'entrée
        Y: labels one-hot
        epochs: nombre d'époques
        lr: learning rate
        batch_size: taille des mini-batches
    """
    n_samples = X.shape[0]
    n_layers = len(dnn.dbn.rbms)
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]
        
        for i in range(0, n_samples, batch_size):
            batch_X = X_shuffled[i:min(i + batch_size, n_samples)]
            batch_Y = Y_shuffled[i:min(i + batch_size, n_samples)]
            m = batch_X.shape[0]
            
            # Forward pass
            activations, output = entree_sortie_reseau(dnn, batch_X)
            
            # Erreur couche de sortie
            delta = output - batch_Y  # Gradient cross-entropy + softmax
            
            # Mise à jour couche classification
            dnn.W_class -= lr * (activations[-1].T @ delta) / m
            dnn.b_class -= lr * np.mean(delta, axis=0)
            
            # Rétropropagation dans le DBN
            current_delta = delta @ dnn.W_class.T
            
            for l in range(n_layers - 1, -1, -1):
                rbm = dnn.dbn.rbms[l]
                a = activations[l + 1]
                
                # Gradient sigmoïde
                grad = current_delta * a * (1 - a)
                
                # Mise à jour poids et biais
                rbm.W -= lr * (activations[l].T @ grad) / m
                rbm.b -= lr * np.mean(grad, axis=0)
                
                if l > 0:
                    current_delta = grad @ rbm.W.T
        
        # Dans la fonction retropropagation, vers la ligne 208
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            _, pred = entree_sortie_reseau(dnn, X)
            # CORRECTION ICI: remplacer 'output' par 'pred'
            loss = -np.mean(np.sum(Y * np.log(pred + 1e-8), axis=1)) 
            acc = np.mean(np.argmax(pred, axis=1) == np.argmax(Y, axis=1))
            print(f"Epoch {epoch+1} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")
            
    return dnn


def test_DNN(dnn, X_test, y_test):
    """
    Évalue le DNN sur des données de test.
    
    Retourne:
        accuracy: taux de bonne classification
    """
    _, output = entree_sortie_reseau(dnn, X_test)
    predictions = np.argmax(output, axis=1)
    accuracy = np.mean(predictions == y_test)
    return accuracy  

# ============================================================================
# FONCTION D'EXPÉRIMENTATION
# ============================================================================

def run_experiment(X_train, X_test, y_train, y_test, 
                   layer_sizes, n_classes=10,
                   pretrain_epochs=20, finetune_epochs=50,
                   lr_pretrain=0.1, lr_finetune=0.1,
                   batch_size=100, verbose=False):
    """
    Lance une expérience avec et sans pré-entrainement.
    
    Retourne:
        acc_pretrained: accuracy avec pré-entrainement
        acc_random: accuracy sans pré-entrainement
    """
    Y_train = one_hot(y_train, n_classes)
    
    # DNN avec pré-entrainement
    dnn_pre = init_DNN(layer_sizes, n_classes)
    dnn_pre = pretrain_DNN(dnn_pre, X_train, pretrain_epochs, lr_pretrain, batch_size, verbose=verbose)
    dnn_pre = retropropagation(dnn_pre, X_train, Y_train, finetune_epochs, lr_finetune, batch_size, verbose=verbose)
    acc_pretrained = test_DNN(dnn_pre, X_test, y_test)
    
    # DNN sans pré-entrainement (aléatoire)
    dnn_rand = init_DNN(layer_sizes, n_classes)
    dnn_rand = retropropagation(dnn_rand, X_train, Y_train, finetune_epochs, lr_finetune, batch_size, verbose=verbose)
    acc_random = test_DNN(dnn_rand, X_test, y_test)
    
    return acc_pretrained, acc_random

# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("ÉTUDE DNN - CLASSIFICATION MNIST")
    print("Comparaison: pré-entrainement DBN vs initialisation aléatoire")
    print("="*70)
    
    # Paramètres globaux
    N_SAMPLES = 10000  # Pour accélérer (utiliser 70000 pour résultats complets)
    PRETRAIN_EPOCHS = 20
    FINETUNE_EPOCHS = 50
    LR_PRETRAIN = 0.1
    LR_FINETUNE = 0.1
    BATCH_SIZE = 100
    
    # Chargement données
    X_train_full, X_test, y_train_full, y_test = load_mnist_binarized(N_SAMPLES)
    
    # Affichage exemples
    plt.figure(figsize=(12, 2))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(X_train_full[i].reshape(28, 28), cmap='gray')
        plt.title(str(y_train_full[i]))
        plt.axis('off')
    plt.suptitle("Exemples MNIST binarisé")
    plt.show()
    
    # =========================================
    # TEST INITIAL
    # =========================================
    print("\n" + "="*70)
    print("TEST INITIAL: Architecture 784 -> 500 -> 10")
    print("="*70)
    
    layer_sizes = [784, 500]
    acc_pre, acc_rand = run_experiment(
        X_train_full, X_test, y_train_full, y_test,
        layer_sizes, verbose=True
    )
    print(f"\nRésultats:")
    print(f"  Avec pré-entrainement: {acc_pre:.4f}")
    print(f"  Sans pré-entrainement: {acc_rand:.4f}")
    
    # =========================================
    # FIGURE 1: IMPACT DE LA PROFONDEUR
    # =========================================
    print("\n" + "="*70)
    print("FIGURE 1: Erreur de classification en fonction de la profondeur")
    print("="*70)
    
    depths = [
        [784, 200, 200],                # 2 couches (hidden)
        [784, 200, 200, 200],           # 3 couches
        [784, 200, 200, 200, 200],      # 4 couches
        [784, 200, 200, 200, 200, 200], # 5 couches
    ]
    depth_labels = ["2 couches", "3 couches", "4 couches", "5 couches"]
    
    acc_pre_depth = []
    acc_rand_depth = []
    
    for arch, name in zip(depths, depth_labels):
        print(f"Test: {name} - {arch}")
        acc_pre, acc_rand = run_experiment(
            X_train_full, X_test, y_train_full, y_test, arch
        )
        acc_pre_depth.append(acc_pre)
        acc_rand_depth.append(acc_rand)
        print(f"  Pré-entrainé: {acc_pre:.4f}, Aléatoire: {acc_rand:.4f}")
    
    # Conversion en taux d'erreur
    err_pre_depth = [1 - a for a in acc_pre_depth]
    err_rand_depth = [1 - a for a in acc_rand_depth]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(depths))
    width = 0.35
    
    plt.bar(x - width/2, err_pre_depth, width, label='Pré-entrainé (DBN)', color='blue', alpha=0.7)
    plt.bar(x + width/2, err_rand_depth, width, label='Aléatoire', color='orange', alpha=0.7)
    
    plt.xlabel("Profondeur du réseau", fontsize=12)
    plt.ylabel("Taux d'erreur", fontsize=12)
    plt.title("Figure 1: Impact de la profondeur sur le taux d'erreur", fontsize=14)
    plt.xticks(x, depth_labels)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "figure1_profondeur.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    # =========================================
    # FIGURE 2: IMPACT DE LA LARGEUR
    # =========================================
    print("\n" + "="*70)
    print("FIGURE 2: Erreur de classification en fonction de la largeur")
    print("="*70)
    
    widths = [
        [784, 100, 100],
        [784, 300, 300],
        [784, 500, 500],
        [784, 700, 700],
    ]
    width_labels = ["100", "300", "500", "700"]
    
    acc_pre_width = []
    acc_rand_width = []
    
    for arch, name in zip(widths, width_labels):
        print(f"Test: largeur={name} - {arch}")
        acc_pre, acc_rand = run_experiment(
            X_train_full, X_test, y_train_full, y_test, arch
        )
        acc_pre_width.append(acc_pre)
        acc_rand_width.append(acc_rand)
        print(f"  Pré-entrainé: {acc_pre:.4f}, Aléatoire: {acc_rand:.4f}")
    
    err_pre_width = [1 - a for a in acc_pre_width]
    err_rand_width = [1 - a for a in acc_rand_width]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(widths))
    
    plt.bar(x - width/2, err_pre_width, 0.35, label='Pré-entrainé (DBN)', color='blue', alpha=0.7)
    plt.bar(x + 0.35/2, err_rand_width, 0.35, label='Aléatoire', color='orange', alpha=0.7)
    
    plt.xlabel("Nombre de neurones cachés", fontsize=12)
    plt.ylabel("Taux d'erreur", fontsize=12)
    plt.title("Figure 2: Impact de la largeur sur le taux d'erreur", fontsize=14)
    plt.xticks(x, width_labels)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "figure2_largeur.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    # =========================================
    # FIGURE 3: IMPACT DE LA TAILLE DES DONNÉES
    # =========================================
    print("\n" + "="*70)
    print("FIGURE 3: Erreur de classification en fonction de la taille des données")
    print("="*70)
    
    data_sizes = [500, 1000, 2000, 5000, 8000]
    
    acc_pre_data = []
    acc_rand_data = []
    
    for n in data_sizes:
        print(f"Test: n={n} données d'entrainement")
        X_sub = X_train_full[:n]
        y_sub = y_train_full[:n]
        
        acc_pre, acc_rand = run_experiment(
            X_sub, X_test, y_sub, y_test, [784, 500]
        )
        acc_pre_data.append(acc_pre)
        acc_rand_data.append(acc_rand)
        print(f"  Pré-entrainé: {acc_pre:.4f}, Aléatoire: {acc_rand:.4f}")
    
    err_pre_data = [1 - a for a in acc_pre_data]
    err_rand_data = [1 - a for a in acc_rand_data]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(data_sizes, err_pre_data, 'o-', label='Pré-entrainé (DBN)', 
             color='blue', linewidth=2, markersize=8)
    plt.plot(data_sizes, err_rand_data, 's-', label='Aléatoire', 
             color='orange', linewidth=2, markersize=8)
    
    plt.xlabel("Nombre de données d'entrainement", fontsize=12)
    plt.ylabel("Taux d'erreur", fontsize=12)
    plt.title("Figure 3: Impact de la taille des données sur le taux d'erreur", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "figure3_donnees.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    # =========================================
    # RÉSUMÉ
    # =========================================
    print("\n" + "="*70)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*70)
    
    print("\nFigure 1 - Profondeur:")
    for name, pre, rand in zip(depth_labels, err_pre_depth, err_rand_depth):
        print(f"  {name}: erreur pré-entrainé={pre:.4f}, aléatoire={rand:.4f}")
    
    print("\nFigure 2 - Largeur:")
    for name, pre, rand in zip(width_labels, err_pre_width, err_rand_width):
        print(f"  {name} neurones: erreur pré-entrainé={pre:.4f}, aléatoire={rand:.4f}")
    
    print("\nFigure 3 - Taille des données:")
    for n, pre, rand in zip(data_sizes, err_pre_data, err_rand_data):
        print(f"  {n} exemples: erreur pré-entrainé={pre:.4f}, aléatoire={rand:.4f}")
    
    print("\n" + "="*70)
    print("CONCLUSIONS:")
    print("- Le pré-entrainement DBN aide surtout avec peu de données")
    print("- L'avantage diminue avec l'augmentation des données")
    print("- Les réseaux profonds bénéficient plus du pré-entrainement")
    print("="*70)
