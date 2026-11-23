import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import X, y
from decision_tree import ArbreDecision


def test_classification_gini():
    a = ArbreDecision("gini").fit(X, y)
    assert a.score(X, y) == 1.0


def test_entropie():
    a = ArbreDecision("entropie", max_depth=3).fit(X, y)
    assert a.score(X, y) > 0.8


def test_regression_mse():
    yr = X[:, 0] * 2.0
    a = ArbreDecision("mse", max_depth=4).fit(X, yr)
    assert a.score(X, yr) > 0.9


if __name__ == "__main__":
    test_classification_gini(); test_entropie(); test_regression_mse()
    print("OK")
