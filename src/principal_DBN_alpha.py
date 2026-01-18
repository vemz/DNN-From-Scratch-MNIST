import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results', 'DBN')
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

def lire_alpha_digit(indices=None):
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
    def __init__(self, W, a, b):
        self.W = W
        self.a = a
        self.b = b


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def init_RBM(p, q):
    W = np.random.normal(0, 0.01, (p, q))
    a = np.zeros(p)
    b = np.zeros(q)
    return RBM(W, a, b)


def entree_sortie_RBM(rbm, X):
    return sigmoid(X @ rbm.W + rbm.b)


def sortie_entree_RBM(rbm, H):
    return sigmoid(H @ rbm.W.T + rbm.a)


def train_RBM(rbm, X, epochs, lr, batch_size, k=1, verbose=True):
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
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs} - Erreur reconstruction: {avg_error:.4f}")
    
    return rbm, errors


def generer_image_RBM(rbm, n_iter_gibbs, n_images, image_shape=(20, 16), show=True):
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


class DBN:
    def __init__(self, rbms):
        self.rbms = rbms


def init_DBN(layer_sizes):
    rbms = []
    for i in range(len(layer_sizes) - 1):
        rbm = init_RBM(layer_sizes[i], layer_sizes[i+1])
        rbms.append(rbm)
    return DBN(rbms)


def train_DBN(dbn, X, epochs, lr, batch_size, k=1, verbose=True):
    n_rbms = len(dbn.rbms)
    current_input = X.copy()
    all_errors = []
    
    for i, rbm in enumerate(dbn.rbms):
        if verbose:
            print(f"\n--- Entrainement RBM couche {i+1}/{n_rbms} ---")
            print(f"    Dimensions: {rbm.W.shape[0]} -> {rbm.W.shape[1]}")
        
        rbm, errors = train_RBM(rbm, current_input, epochs, lr, batch_size, k, verbose)
        all_errors.append(errors)
        rbm, errors = train_RBM(rbm, current_input, epochs, lr, batch_size, k, verbose)
        probs = entree_sortie_RBM(rbm, current_input)
        current_input = (np.random.rand(*probs.shape) < probs).astype(float)
    
    return dbn, all_errors


def generer_image_DBN(dbn, n_iter_gibbs, n_images, image_shape=(20, 16), show=True):
    top_rbm = dbn.rbms[-1]
    q_top = top_rbm.W.shape[1]
    
    h = (np.random.rand(n_images, q_top) < 0.5).astype(float)
    
    for _ in range(n_iter_gibbs):
        v_prob = sortie_entree_RBM(top_rbm, h)
        v = (np.random.rand(n_images, top_rbm.W.shape[0]) < v_prob).astype(float)
        h_prob = entree_sortie_RBM(top_rbm, v)
        h = (np.random.rand(n_images, q_top) < h_prob).astype(float)
    
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


if __name__ == "__main__":
    
    EPOCHS = 100
    LR = 0.1
    BATCH_SIZE = 10
    N_GIBBS = 200
    
    print(f"\nParamètres: epochs={EPOCHS}, lr={LR}, batch={BATCH_SIZE}")
    
    alphabet_indices = list(range(10, 15))
    X = lire_alpha_digit(alphabet_indices)
    print(f"Données: {X.shape[0]} images, {X.shape[1]} pixels (20x16)")
    
    plt.figure(figsize=(15, 4))
    random_indices = np.random.choice(X.shape[0], 5, replace=False)
    for i, idx in enumerate(random_indices):
        plt.subplot(1, 5, i+1)
        plt.imshow(X[idx].reshape(20, 16), cmap='gray')
        plt.axis('off')
    plt.suptitle("Exemples de données d'entrainement (aléatoire)")
    plt.show()
    
    print("Test: 1 couche - [320, 200]")
    dbn1 = init_DBN([320, 200])
    dbn1, _ = train_DBN(dbn1, X, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE)

    print("Test: 2 couches - [320, 200, 100]")
    dbn2 = init_DBN([320, 200, 100])
    dbn2, all_errors = train_DBN(dbn2, X, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE)
    
    print("Test: 3 couches - [320, 200, 100, 50]")
    dbn3 = init_DBN([320, 200, 100, 50])
    dbn3, _ = train_DBN(dbn3, X, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE)
    
    print("Test: 4 couches - [320, 200, 100, 50, 25]")
    dbn4 = init_DBN([320, 200, 100, 50, 25])
    dbn4, _ = train_DBN(dbn4, X, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE)

    configs = [
        ("1 couche\n[320, 200]", dbn1),
        ("2 couches\n[320, 200, 100]", dbn2),
        ("3 couches\n[320, 200, 100, 50]", dbn3),
        ("4 couches\n[320, 200, 100, 50, 25]", dbn4)
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, (name, model) in enumerate(configs):
        imgs = generer_image_DBN(model, n_iter_gibbs=N_GIBBS, n_images=1, show=False)
        axes[idx].imshow(imgs[0].reshape(20, 16), cmap='gray')
        axes[idx].set_title(name)
        axes[idx].axis('off')

    plt.suptitle("Impact de la profondeur du DBN", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "analyse_dbn_profondeur.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Analyse 2: DBN sur plusieurs caractères")
    
    char_sets = [[10], [10, 11, 12], [10, 11, 12, 13, 14]]
    char_names = ["A seul", "A-C", "A-E (5 lettres)"]
    
    fig, axes = plt.subplots(len(char_sets), 5, figsize=(12, 3*len(char_sets)))
    
    for row, (chars, name) in enumerate(zip(char_sets, char_names)):
        print(f"\nApprentissage sur: {name}")
        X_multi = lire_alpha_digit(chars)
        print(f"  {X_multi.shape[0]} images")
        
        dbn_test = init_DBN([320, 200, 100])
        dbn_test, _ = train_DBN(dbn_test, X_multi, epochs=300, lr=LR, batch_size=BATCH_SIZE, verbose=False)
        imgs = generer_image_DBN(dbn_test, N_GIBBS, 5, show=False)
        
        for col in range(5):
            axes[row, col].imshow(imgs[col].reshape(20, 16), cmap='gray')
            axes[row, col].axis('off')
            
        axes[row, 0].set_ylabel(name, fontsize=12, rotation=0, labelpad=40, ha='right')
        axes[row, 0].axis('on')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 0].spines['top'].set_visible(False)
        axes[row, 0].spines['right'].set_visible(False)
        axes[row, 0].spines['bottom'].set_visible(False)
        axes[row, 0].spines['left'].set_visible(False)
    
    plt.suptitle("Génération DBN selon le nombre de caractères appris", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "analyse_dbn_caracteres.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Comparaison RBM (1 couche) vs DBN (2 couches)")
    
    X_multi = lire_alpha_digit([10, 11, 12, 13, 14])
    
    print("\nEntraînement RBM simple (320 -> 200)...")
    rbm_simple = init_RBM(320, 200)
    rbm_simple, _ = train_RBM(rbm_simple, X_multi, epochs=100, lr=LR, batch_size=BATCH_SIZE, verbose=False)
    rbm_imgs = generer_image_RBM(rbm_simple, N_GIBBS, 5, show=False)
    
    print("Entraînement DBN (320 -> 200 -> 100)...")
    dbn_compare = init_DBN([320, 200, 100])
    dbn_compare, _ = train_DBN(dbn_compare, X_multi, epochs=100, lr=LR, batch_size=BATCH_SIZE, verbose=False)
    dbn_imgs = generer_image_DBN(dbn_compare, N_GIBBS, 5, show=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(5):
        axes[0, i].imshow(rbm_imgs[i].reshape(20, 16), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(dbn_imgs[i].reshape(20, 16), cmap='gray')
        axes[1, i].axis('off')
    
    for row, name in enumerate(["RBM", "DBN"]):
        axes[row, 0].set_ylabel(name, fontsize=12, rotation=0, labelpad=40, ha='right')
        axes[row, 0].axis('on')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 0].spines['top'].set_visible(False)
        axes[row, 0].spines['right'].set_visible(False)
        axes[row, 0].spines['bottom'].set_visible(False)
        axes[row, 0].spines['left'].set_visible(False)
    
    plt.suptitle("Comparaison RBM vs DBN sur caractères A-E", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparaison_rbm_dbn.png"), dpi=150, bbox_inches='tight')
    plt.show()
