# TP Deep Neural Networks
## Telecom SudParis - MAT5016

Ce projet implémente un réseau de neurones profond pré-entrainé pour la classification de chiffres manuscrits, comparant les performances d'un réseau pré-entrainé avec un réseau initialisé aléatoirement.

---

## Structure du projet

```
MAT5016/
├── principal_RBM_alpha.py    # Script autonome: RBM sur Binary AlphaDigits
├── principal_DBN_alpha.py    # Script autonome: DBN sur Binary AlphaDigits
├── principal_DNN_MNIST.py    # Script autonome: DNN sur MNIST (3 figures comparatives)
├── binaryalphadigs.mat       # Base de données Binary AlphaDigits
└── README.md                 # Ce fichier
```

**Note:** Chaque script `principal_*.py` est **100% autonome** - tout le code (RBM, DBN, DNN) est inclus dans chaque fichier sans imports externes.

---

## Installation des dépendances

```bash
pip install numpy matplotlib scipy scikit-learn
```

---

## Données requises

### Binary AlphaDigits
Le fichier `binaryalphadigs.mat` est inclus dans le projet.  
Source originale : http://www.cs.nyu.edu/~roweis/data.html (via Web Archive)

### MNIST
La base MNIST est téléchargée automatiquement via `scikit-learn` lors de la première exécution.

---

## Exécution des scripts

### 1. Étude préliminaire RBM (Binary AlphaDigits)

```bash
python principal_RBM_alpha.py
```

Ce script :
- Entraine un RBM sur les caractères sélectionnés
- Génère des images via l'échantillonneur de Gibbs
- Analyse l'impact du nombre de neurones cachés
- Analyse le pouvoir modélisant selon le nombre de caractères

**Figures générées :**
- `analyse_rbm_neurones.png`
- `analyse_rbm_caracteres.png`

---

### 2. Étude préliminaire DBN (Binary AlphaDigits)

```bash
python principal_DBN_alpha.py
```

Ce script :
- Entraine un DBN avec la procédure Greedy Layer-Wise
- Génère des images à partir du DBN
- Analyse l'impact de la profondeur
- Compare RBM vs DBN

**Figures générées :**
- `analyse_dbn_profondeur.png`
- `analyse_dbn_caracteres.png`
- `comparaison_rbm_dbn.png`

---

### 3. Étude comparative sur MNIST

```bash
python principal_DNN_MNIST.py
```

Ce script réalise l'étude comparative principale :

**Figure 1** : Erreur vs Profondeur (nombre de couches)

**Figure 2** : Erreur vs Largeur (neurones par couche)

**Figure 3** : Erreur vs Données d'entrainement

**Figures générées :**
- `figure1_profondeur.png`
- `figure2_largeur.png`
- `figure3_donnees.png`

---

## Fonctions implémentées

### RBM (Restricted Boltzmann Machine)
- `init_RBM(p, q)` : initialise un RBM (poids ~ N(0, 0.01), biais = 0)
- `entree_sortie_RBM(rbm, X)` : calcule P(h=1|v)
- `sortie_entree_RBM(rbm, H)` : calcule P(v=1|h)
- `train_RBM(rbm, X, epochs, lr, batch_size, k)` : apprentissage CD-k
- `generer_image_RBM(rbm, n_iter_gibbs, n_images)` : génération par Gibbs

### DBN (Deep Belief Network)
- `init_DBN(layer_sizes)` : initialise un DBN (liste de RBMs)
- `train_DBN(dbn, X, epochs, lr, batch_size)` : Greedy Layer-Wise training
- `generer_image_DBN(dbn, n_iter_gibbs, n_images)` : génération top-down

### DNN (Deep Neural Network)
- `init_DNN(layer_sizes, n_classes)` : initialise un DNN
- `pretrain_DNN(dnn, X, epochs, lr, batch_size)` : pré-entrainement DBN
- `calcul_softmax(z)` : fonction softmax
- `entree_sortie_reseau(dnn, X)` : forward pass complet
- `retropropagation(dnn, X, Y, epochs, lr, batch_size)` : fine-tuning supervisé
- `test_DNN(dnn, X_test, y_test)` : évaluation accuracy

---

## Hyperparamètres recommandés

| Paramètre | RBM/DBN | DNN (rétropropagation) |
|-----------|---------|------------------------|
| Epochs | 100 | 50 |
| Learning rate | 0.1 | 0.1 |
| Batch size | 10 (AlphaDigits) / 100 (MNIST) | 100 |
| Gibbs iterations | 100-200 | - |

---

## Résultats attendus

L'avantage du pré-entrainement DBN est plus marqué :
- Avec peu de données labellisées
- Avec des réseaux plus profonds
- En début d'entrainement (convergence plus rapide)

---

## Installation

```bash
pip install numpy scipy matplotlib scikit-learn pandas
```

---

## Auteur

TP réalisé dans le cadre du cours MAT5016 - Deep Learning  
Telecom SudParis - Décembre 2025
