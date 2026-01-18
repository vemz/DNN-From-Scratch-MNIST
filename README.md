# Deep Neural Networks - TP MAT5016

Implémentation de RBM, DBN et DNN en NumPy (sans framework de Deep Learning).

## Installation

```bash
pip install -r requirements.txt
```

## Lancer les scripts

Tous les scripts se trouvent dans `src/` :

### 1. RBM sur Binary AlphaDigits
```bash
python src/principal_RBM_alpha.py
```
Produit 2 graphiques dans `results/RBM/`
- Analyse de l'impact du nombre de neurones cachés
- Analyse du nombre de caractères appris

### 2. DBN sur Binary AlphaDigits
```bash
python src/principal_DBN_alpha.py
```
Produit 3 graphiques dans `results/DBN/`
- Impact de la profondeur
- Impact du nombre de caractères
- Comparaison RBM vs DBN

### 3. DNN sur MNIST (comparaison pré-entrainé vs aléatoire)
```bash
python src/principal_DNN_MNIST.py
```
Produit 3 figures dans `results/DNN/`
- Figure 1 : Erreur vs profondeur (2-5 couches de 200)
- Figure 2 : Erreur vs largeur (100-700 neurones, 2 couches)
- Figure 3 : Erreur vs taille des données (1000-60000 exemples)

---

Telecom SudParis - MAT5016 - Janvier 2026