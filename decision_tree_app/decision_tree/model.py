import numpy as np

from .critere import IMPURETE
from .construction import construire, descendre

# "mse" est accepté comme alias de "variance" (régression)
ALIAS = {"mse": "variance"}


class ArbreDecision:
    def __init__(self, critere="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_gain=0.0):
        critere = ALIAS.get(critere, critere)
        if critere not in IMPURETE:
            raise ValueError(f"Critère inconnu : {critere!r}. "
                             f"Choix : {list(IMPURETE)} (ou 'mse').")
        self.critere = critere
        self.params = {"max_depth": max_depth,
                       "min_samples_split": min_samples_split,
                       "min_samples_leaf": min_samples_leaf,
                       "min_gain": min_gain}
        self.racine = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.regression = (self.critere == "variance")
        if self.regression:
            y = y.astype(float)
        self.racine = construire(X, y, IMPURETE[self.critere],
                                 self.regression, 0, self.params)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([descendre(self.racine, x).prediction for x in X])

    def score(self, X, y):
        y = np.asarray(y)
        yp = self.predict(X)
        if self.regression:  # coefficient R^2
            ss_tot = np.sum((y - y.mean()) ** 2)
            return float(1 - np.sum((y - yp) ** 2) / ss_tot) if ss_tot > 0 else 1.0
        return float(np.mean(yp == y))  # exactitude
