def profondeur(noeud):
    if noeud.est_feuille():
        return 0
    return 1 + max(profondeur(noeud.gauche), profondeur(noeud.droit))


def nb_feuilles(noeud):
    if noeud.est_feuille():
        return 1
    return nb_feuilles(noeud.gauche) + nb_feuilles(noeud.droit)
