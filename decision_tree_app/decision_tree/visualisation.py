def afficher(noeud, noms, indent=""):
    if noeud.est_feuille():
        if noeud.repartition is not None:
            print(f"{indent}-> classe {noeud.prediction} "
                  f"(n={noeud.n}, {noeud.repartition})")
        else:
            print(f"{indent}-> valeur {noeud.prediction:.2f} (n={noeud.n})")
    else:
        print(f"{indent}si {noms[noeud.attribut]} <= {noeud.seuil:.2f} :")
        afficher(noeud.gauche, noms, indent + "    ")
        print(f"{indent}sinon :")
        afficher(noeud.droit, noms, indent + "    ")


def regles(noeud, noms, etiquettes=None, conditions=()):
    if noeud.est_feuille():
        cond = " ET ".join(conditions) if conditions else "(toujours)"
        sortie = noeud.prediction
        if etiquettes:
            sortie = etiquettes.get(sortie, sortie)
        if isinstance(sortie, float):
            sortie = f"{sortie:.2f}"
        return [f"SI {cond} ALORS {sortie}"]
    nom = noms[noeud.attribut]
    v = f"{noeud.seuil:.2f}"
    return (regles(noeud.gauche, noms, etiquettes, conditions + (f"{nom} <= {v}",)) +
            regles(noeud.droit, noms, etiquettes, conditions + (f"{nom} > {v}",)))
