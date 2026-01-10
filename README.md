# Deep Neural Networks - From Scratch

> Implémentation complète de RBM, DBN et DNN en pure Python / Numpy, sans framework de Deep Learning.

Ce projet a été réalisé dans le cadre du cours de Deep Learning (MAT5016) à Télécom SudParis. Il vise à comprendre les mécanismes internes des modèles génératifs et discriminatifs profonds.

---

## Fonctionnalités

Le projet implémente les architectures suivantes de zéro :

*   **RBM (Restricted Boltzmann Machine)** :
    *   Apprentissage non-supervisé par *Contrastive Divergence* (CD-k).
    *   Génération d'images par échantillonnage de Gibbs.
*   **DBN (Deep Belief Network)** :
    *   Empilement de RBMs.
    *   Entrainement *Greedy Layer-Wise*.
*   **DNN (Deep Neural Network)** :
    *   Utilisation du DBN pour pré-initialiser le réseau.
    *   Fine-tuning supervisé par rétropropagation (Backpropagation).
    *   Comparaison : *Pre-trained* vs *Random Initialization*.

## Structure du Projet

Les dossiers `data/` et `results/` sont exclus du contrôle de version (gitignore) pour ne garder que le code source léger. Vous devrez les créer ou les peupler localement comme indiqué ci-dessous.

```bash
.
├── src/                # Code source (RBM, DBN, DNN)
├── data/               # Dossier local pour les datasets (à créer)
├── results/            # Dossier local pour les graphiques générés (organisé par modèle)
├── docs/               # Sujet et rapport
└── README.md           # Documentation
```

## Installation et Configuration

1.  **Cloner le dépôt**
    ```bash
    git clone https://github.com/username/DNN-From-Scratch-MNIST.git
    cd DNN-From-Scratch-MNIST
    ```

2.  **Installer les dépendances**
    ```bash
    pip install numpy matplotlib scipy scikit-learn pandas
    ```

3.  **Télécharger les données**
    Le dataset `binaryalphadigs.mat` n'est pas inclus dans le dépôt.
    *   Téléchargez-le depuis Kaggle : [Binary Alpha Digits Dataset](https://www.kaggle.com/datasets/angevalli/binary-alpha-digits)
    *   Créez le dossier `data` et placez le fichier `binaryalphadigs.mat` dedans :
        ```bash
        mkdir -p data
        mv /chemin/vers/binaryalphadigs.mat data/
        ```
    *(Note : Le dataset MNIST sera téléchargé automatiquement par scikit-learn lors de la première exécution)*

## Utilisation

Les scripts doivent être lancés depuis la racine du projet.

### 1. Restricted Boltzmann Machine (RBM)
Entraîne un RBM sur des caractères (AlphaDigits) et génère de nouvelles formes.
```bash
python src/principal_RBM_alpha.py
```
*Les graphiques générés seront sauvegardés dans le dossier `results/RBM/`.*

### 2. Deep Belief Network (DBN)
Entraîne un réseau profond couche par couche.
```bash
python src/principal_DBN_alpha.py
```
*Les graphiques générés seront sauvegardés dans le dossier `results/DBN/`.*

### 3. Classification MNIST (DNN)
Compare les performances avec et sans pré-entrainement.
```bash
python src/principal_DNN_MNIST.py
```
*Les courbes comparatives seront sauvegardées dans le dossier `results/DNN/`.*

## Résultats principaux

L'étude démontre que le pré-entrainement non-supervisé (DBN) améliore significativement les performances du réseau de neurones, particulièrement lorsque :
1.  Le nombre de données d'entrainement est limité.
2.  Le réseau est très profond (nombreuses couches cachées).

---

## Auteurs

TP réalisé dans le cadre du cours MAT5016 - Deep Learning
Télécom SudParis - Janvier 2026