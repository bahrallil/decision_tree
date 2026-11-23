from data.dataset import X, y, noms
from decision_tree import ArbreDecision, afficher, regles, profondeur, nb_feuilles

arbre = ArbreDecision(critere="gini", max_depth=3, min_samples_split=2)
arbre.fit(X, y)

print("arbre de decision")
afficher(arbre.racine, noms)
print()
print("profondeur :", profondeur(arbre.racine), "| feuilles :", nb_feuilles(arbre.racine))
print("exactitude :", arbre.score(X, y))
print()
for r in regles(arbre.racine, noms):
    print(r)
