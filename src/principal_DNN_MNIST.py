import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results', 'DNN')
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

def load_mnist_binarized(n_samples=None, threshold=0.5):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(float) / 255.0
    y = mnist.target.astype(int)
    
    X = (X > threshold).astype(float)
    
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"MNIST: {X_train.shape[0]} train, {X_test.shape[0]} test, {X.shape[1]} pixels")
    return X_train, X_test, y_train, y_test


def one_hot(y, n_classes=10):
    n = len(y)
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1
    return Y

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


def train_RBM(rbm, X, epochs, lr, batch_size, k=1, verbose=False):
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

def init_DBN(layer_sizes):
    rbms = []
    for i in range(len(layer_sizes) - 1):
        rbm = init_RBM(layer_sizes[i], layer_sizes[i+1])
        rbms.append(rbm)
    return rbms

class DNN:
    def __init__(self, rbms, W_class, b_class):
        self.rbms = rbms
        self.W_class = W_class
        self.b_class = b_class

def init_DNN(layer_sizes, n_classes=10):
    rbms = init_DBN(layer_sizes) 
    
    last_hidden = layer_sizes[-1]
    W_class = np.random.normal(0, 0.01, (last_hidden, n_classes))
    b_class = np.zeros(n_classes)
    return DNN(rbms, W_class, b_class)


def pretrain_DNN(dnn, X, epochs, lr, batch_size, k=1, verbose=False):
    current_input = X.copy()
    for i, rbm in enumerate(dnn.rbms):
        if verbose:
            print(f"  Pretrain couche {i+1}/{len(dnn.rbms)}")
        train_RBM(rbm, current_input, epochs, lr, batch_size, k, verbose)
        current_input = entree_sortie_RBM(rbm, current_input)
    return dnn


def calcul_softmax(z):
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def entree_sortie_reseau(dnn, X):
    activations = [X]
    current = X
    
    for rbm in dnn.rbms:
        current = entree_sortie_RBM(rbm, current)
        activations.append(current)
    
    logits = current @ dnn.W_class + dnn.b_class
    output = calcul_softmax(logits)
    
    return activations, output


def retropropagation(dnn, X, Y, epochs, lr, batch_size, verbose=False):
    n_samples = X.shape[0]
    n_layers = len(dnn.rbms)
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]
        
        for i in range(0, n_samples, batch_size):
            batch_X = X_shuffled[i:min(i + batch_size, n_samples)]
            batch_Y = Y_shuffled[i:min(i + batch_size, n_samples)]
            m = batch_X.shape[0]
            
            activations, output = entree_sortie_reseau(dnn, batch_X)
            
            delta = output - batch_Y
            
            dnn.W_class -= lr * (activations[-1].T @ delta) / m
            dnn.b_class -= lr * np.mean(delta, axis=0)
            
            current_delta = delta @ dnn.W_class.T
            
            for l in range(n_layers - 1, -1, -1):
                rbm = dnn.rbms[l]
                a = activations[l + 1]
                
                grad = current_delta * a * (1 - a)
                
                rbm.W -= lr * (activations[l].T @ grad) / m
                rbm.b -= lr * np.mean(grad, axis=0)
                
                if l > 0:
                    current_delta = grad @ rbm.W.T
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            _, pred = entree_sortie_reseau(dnn, X)
            loss = -np.mean(np.sum(Y * np.log(pred + 1e-8), axis=1)) 
            acc = np.mean(np.argmax(pred, axis=1) == np.argmax(Y, axis=1))
            print(f"Epoch {epoch+1} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")
            
    return dnn

def test_DNN(dnn, X_test, y_test):
    _, output = entree_sortie_reseau(dnn, X_test)
    predictions = np.argmax(output, axis=1)
    accuracy = np.mean(predictions == y_test)
    return accuracy  

# Expériences

def run_experiment(X_train, X_test, y_train, y_test, 
                   layer_sizes, n_classes=10,
                   pretrain_epochs=100, finetune_epochs=200,
                   lr_pretrain=0.1, lr_finetune=0.1,
                   batch_size=100, verbose=False):
    import copy

    Y_train = one_hot(y_train, n_classes)
    
    dnn_pre = init_DNN(layer_sizes, n_classes)
    dnn_rand = copy.deepcopy(dnn_pre)
    
    dnn_pre = pretrain_DNN(dnn_pre, X_train, pretrain_epochs, lr_pretrain, batch_size, verbose=verbose)
    dnn_pre = retropropagation(dnn_pre, X_train, Y_train, finetune_epochs, lr_finetune, batch_size, verbose=verbose)
    
    dnn_rand = retropropagation(dnn_rand, X_train, Y_train, finetune_epochs, lr_finetune, batch_size, verbose=verbose)
    
    acc_pretrained_test = test_DNN(dnn_pre, X_test, y_test)
    acc_random_test = test_DNN(dnn_rand, X_test, y_test)
    
    return acc_pretrained_test, acc_random_test

if __name__ == "__main__":
    
    N_SAMPLES = 70000
    PRETRAIN_EPOCHS = 100
    FINETUNE_EPOCHS = 200
    LR_PRETRAIN = 0.1
    LR_FINETUNE = 0.1
    BATCH_SIZE = 100
    
    X_train_full, X_test, y_train_full, y_test = load_mnist_binarized(N_SAMPLES)
    
    plt.figure(figsize=(12, 2))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(X_train_full[i].reshape(28, 28), cmap='gray')
        plt.title(str(y_train_full[i]))
        plt.axis('off')
    plt.suptitle("Exemples MNIST binarisé")
    plt.show()
    
    dnn_demo = init_DNN([784, 200], 10)
    dnn_demo = pretrain_DNN(dnn_demo, X_train_full[:5000], 50, LR_PRETRAIN, BATCH_SIZE)
    dnn_demo = retropropagation(dnn_demo, X_train_full[:5000], one_hot(y_train_full[:5000]), 100, LR_FINETUNE, BATCH_SIZE)
    
    _, probs = entree_sortie_reseau(dnn_demo, X_train_full[:5])
    for i in range(5):
        print(f"  Image {i} (label={y_train_full[i]}): probs = {np.round(probs[i], 3)}")
        print(f"    -> Prédiction: {np.argmax(probs[i])}")
    
    print("Test initial: Architecture 784 -> 200 -> 10")
    
    layer_sizes = [784, 200]
    acc_pre, acc_rand = run_experiment(
        X_train_full, X_test, y_train_full, y_test,
        layer_sizes, verbose=True
    )
    print(f"\nRésultats:")
    print(f"  Avec pré-entrainement: {acc_pre:.4f}")
    print(f"  Sans pré-entrainement: {acc_rand:.4f}")
    
    print("Figure 1: Erreur en fonction de la profondeur (couches de 200)")
    
    depths = [
        [784, 200, 200],
        [784, 200, 200, 200],
        [784, 200, 200, 200, 200],
        [784, 200, 200, 200, 200, 200],
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
    
    print("Figure 2: Erreur en fonction de la largeur (2 couches)")
    
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
    
    print("Figure 3: Erreur en fonction du nombre de données (2 couches de 200)")

    data_sizes = [1000, 3000, 7000, 10000, 30000, min(60000, len(X_train_full))]
    
    acc_pre_data = []
    acc_rand_data = []
    
    for n in data_sizes:
        print(f"Test: n={n} données d'entrainement")
        X_sub = X_train_full[:n]
        y_sub = y_train_full[:n]
        
        acc_pre, acc_rand = run_experiment(
            X_sub, X_test, y_sub, y_test, [784, 200, 200]
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

    print("Résumé")
    
    print("\nFigure 1 - Profondeur:")
    for name, pre, rand in zip(depth_labels, err_pre_depth, err_rand_depth):
        print(f"  {name}: erreur pré-entrainé={pre:.4f}, aléatoire={rand:.4f}")
    
    print("\nFigure 2 - Largeur:")
    for name, pre, rand in zip(width_labels, err_pre_width, err_rand_width):
        print(f"  {name} neurones: erreur pré-entrainé={pre:.4f}, aléatoire={rand:.4f}")
    
    print("\nFigure 3 - Taille des données:")
    for n, pre, rand in zip(data_sizes, err_pre_data, err_rand_data):
        print(f"  {n} exemples: erreur pré-entrainé={pre:.4f}, aléatoire={rand:.4f}")

    # On recherche maintenant la meilleure architecture
    best_arch = [784, 500, 500, 500] 
    
    print(f"Architecture : {best_arch}")
    
    dnn_opt = init_DNN(best_arch, 10)
    dnn_opt = pretrain_DNN(dnn_opt, X_train_full, epochs=50, lr=0.1, batch_size=100, verbose=True)
    
    Y_train_full = one_hot(y_train_full, 10)
    dnn_opt = retropropagation(dnn_opt, X_train_full, Y_train_full, epochs=100, lr=0.1, batch_size=100, verbose=True)

    acc_final_train = test_DNN(dnn_opt, X_train_full, y_train_full)
    acc_final_test = test_DNN(dnn_opt, X_test, y_test)
    
    print("\nRésultats finaux optimisés:")
    print(f"  Accuracy Train : {acc_final_train:.4f}")
    print(f"  Accuracy Test  : {acc_final_test:.4f}")