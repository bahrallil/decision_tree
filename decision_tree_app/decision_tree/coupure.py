import numpy as np


def meilleure_coupure(X, y, impurete, min_leaf=1):
    n, d = X.shape
    parent = impurete(y)
    meilleur = (None, None, 0.0)
    for j in range(d):
        valeurs = np.unique(X[:, j])
        if len(valeurs) < 2:
            continue
        seuils = (valeurs[:-1] + valeurs[1:]) / 2
        for s in seuils:
            masque = X[:, j] <= s
            ng = int(masque.sum())
            nd = n - ng
            if ng < min_leaf or nd < min_leaf:
                continue
            imp = (ng / n) * impurete(y[masque]) + (nd / n) * impurete(y[~masque])
            gain = parent - imp
            if gain > meilleur[2] + 1e-12:
                meilleur = (j, float(s), float(gain))
    return meilleur
