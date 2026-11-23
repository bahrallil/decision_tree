import numpy as np

from .noeud import Noeud
from .coupure import meilleure_coupure


def construire(X, y, impurete, regression, profondeur, p):
    n = len(y)
    imp = impurete(y)

    if regression:
        noeud = Noeud(n, imp, float(np.mean(y)))
    else:
        classes, c = np.unique(y, return_counts=True)
        noeud = Noeud(n, imp, classes[np.argmax(c)].item(),
                      dict(zip(classes.tolist(), c.tolist())))

    # conditions d'arrêt
    if (imp <= 1e-12 or n < p["min_samples_split"]
            or (p["max_depth"] is not None and profondeur >= p["max_depth"])):
        return noeud

    j, s, gain = meilleure_coupure(X, y, impurete, p["min_samples_leaf"])
    if j is None or gain <= p["min_gain"]:
        return noeud

    masque = X[:, j] <= s
    noeud.attribut, noeud.seuil = j, s
    prof = profondeur + 1

    noeud.gauche = construire(X[masque], y[masque], impurete, regression, prof, p)
    noeud.droit = construire(X[~masque], y[~masque], impurete, regression, prof, p)
    return noeud


def descendre(noeud, x):
    while not noeud.est_feuille():
        noeud = noeud.gauche if x[noeud.attribut] <= noeud.seuil else noeud.droit
    return noeud
