# Deep Neural Networks - From scratch

> Implémentation complète de RBM, DBN et DNN en pure Python / Numpy, sans framework de Deep Learning.

Ce projet a été réalisé dans le cadre du cours de Deep Learning (MAT5016) à Télécom SudParis. Il vise à comprendre les mécanismes internes des modèles génératifs et discriminatifs profonds.

---

## Fonctionnalités

Le projet implémente les architectures suivantes :

*   **RBM (Restricted Boltzmann Machine)** :
    *   Apprentissage non-supervisé par Contrastive Divergence (CD-k).
    *   Génération d'images par échantillonnage de Gibbs.
*   **DBN (Deep Belief Network)** :
    *   Empilement de RBMs.
    *   Entraînement Greedy Layer-Wise.
*   **DNN (Deep Neural Network)** :
    *   Utilisation du DBN pour pré-initialiser le réseau.
    *   Fine-tuning supervisé par rétropropagation (Backpropagation).
    *   Comparaison : Pre-trained vs Random Initialization.

## Structure du projet

Vous devrez créer le dossier `data/` localement comme indiqué ci-dessous. Le dossier 'results/' sera créé automatiquement.

```bash
.
├── src/                # Code source (RBM, DBN, DNN)
├── data/               # Dossier local pour les datasets (à créer)
├── results/            # Dossier local pour les graphiques générés (organisé par modèle)
└── README.md           # Documentation
```

## Installation et configuration

1.  **Cloner le dépôt**
    ```bash
    git clone https://github.com/username/DNN-From-Scratch-MNIST.git
    cd DNN-From-Scratch-MNIST
    ```

2.  **Installer les dépendances**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Télécharger les données** \\
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

L'analyse comparative entre un DNN initialisé aléatoirement et un DNN pré-entraîné (DBN) met en évidence trois phénomènes majeurs :

1.  **Impact de la taille du jeu de données** :
    *   Le pré-entraînement offre un gain de performance critique lorsque les données labellisées sont rares (ex: < 2000 images).
    *   Le DBN agit comme un régularisateur, exploitant la structure des données pour éviter l'overfitting.

2.  **Impact de la profondeur du réseau** :
    *   Pour les architectures profondes (3 couches cachées et plus), l'initialisation aléatoire échoue souvent à converger correctement (problème du vanishing gradient).
    *   Le pré-entraînement initialise les poids dans une région optimale de l'espace des paramètres, permettant l'entraînement efficace de réseaux profonds.

3.  **Qualité des représentations** :
    *   Les filtres appris par le RBM/DBN ressemblent à des détecteurs de traits (bords, boucles), contrairement au bruit typique d'une initialisation aléatoire.


---

## Auteurs

TP réalisé dans le cadre du cours MAT5016 - Deep Learning \\
Télécom SudParis - Janvier 2026