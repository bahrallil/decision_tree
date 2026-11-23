import numpy as np


def gini(y):
    n = len(y)
    if n == 0:
        return 0.0
    _, c = np.unique(y, return_counts=True)
    p = c / n
    return float(1.0 - np.sum(p ** 2))


def entropie(y):
    n = len(y)
    if n == 0:
        return 0.0
    _, c = np.unique(y, return_counts=True)
    p = c / n
    return float(-np.sum(p * np.log2(p)))


def variance(y):
    return float(np.var(y)) if len(y) > 0 else 0.0


IMPURETE = {"gini": gini, "entropie": entropie, "variance": variance}
