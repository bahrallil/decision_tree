import plotly.graph_objects as go

from .metrics import profondeur, nb_feuilles

PALETTE = ["#16A34A", "#F59E0B", "#DC2626", "#9333EA", "#0891B2",
           "#DB2777", "#65A30D", "#EA580C"]
COULEUR_NOEUD = "#2563EB"
COULEUR_REGRESSION = "#16A34A"


def _classes(root):
    """Liste ordonnée des classes présentes dans l'arbre (classification)."""
    vues = []

    def parcours(n):
        if n.repartition:
            for c in n.repartition:
                if c not in vues:
                    vues.append(c)
        if not n.est_feuille():
            parcours(n.gauche)
            parcours(n.droit)

    parcours(root)
    return sorted(vues, key=str)


def visualize_tree(root, feature_names, criterion="gini"):
    """Construit une figure Plotly de l'arbre.

    Les feuilles sont alignées de gauche à droite, chaque parent est centré
    au-dessus de ses enfants : aucun chevauchement, quelle que soit la taille.
    """
    classes = _classes(root)
    couleur_classe = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}

    noeuds = []
    aretes = []
    compteur = [0]

    def traverse(node, depth):
        idx = len(noeuds)
        noeuds.append(None)

        if node.est_feuille():
            x = float(compteur[0])
            compteur[0] += 1
            if node.repartition is not None:
                texte = f"<b>Classe {node.prediction}</b><br>n = {node.n}"
                survol = (f"Classe prédite : {node.prediction}<br>n = {node.n}"
                          f"<br>Répartition : {node.repartition}"
                          f"<br>{criterion} = {node.impurete:.3f}")
                couleur = couleur_classe.get(node.prediction, COULEUR_NOEUD)
            else:
                texte = f"<b>Valeur {node.prediction:.2f}</b><br>n = {node.n}"
                survol = (f"Valeur prédite : {node.prediction:.3f}<br>n = {node.n}"
                          f"<br>variance = {node.impurete:.3f}")
                couleur = COULEUR_REGRESSION
        else:
            g = traverse(node.gauche, depth + 1)
            d = traverse(node.droit, depth + 1)
            x = (noeuds[g]["x"] + noeuds[d]["x"]) / 2
            nom = feature_names[node.attribut]
            texte = (f"<b>{nom} ≤ {node.seuil:.2f}</b>"
                     f"<br>{criterion} = {node.impurete:.3f}<br>n = {node.n}")
            survol = texte.replace("<b>", "").replace("</b>", "")
            couleur = COULEUR_NOEUD
            aretes.append((idx, g, "oui"))
            aretes.append((idx, d, "non"))

        noeuds[idx] = {"x": x, "y": -depth, "texte": texte,
                       "survol": survol, "couleur": couleur}
        return idx

    traverse(root, 0)

    fig = go.Figure()

    ex, ey = [], []
    for p, c, _ in aretes:
        ex += [noeuds[p]["x"], noeuds[c]["x"], None]
        ey += [noeuds[p]["y"], noeuds[c]["y"], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                             line=dict(color="#94A3B8", width=2),
                             hoverinfo="skip"))

    for p, c, lab in aretes:
        fig.add_annotation(
            x=(noeuds[p]["x"] + noeuds[c]["x"]) / 2,
            y=(noeuds[p]["y"] + noeuds[c]["y"]) / 2,
            text=lab, showarrow=False,
            font=dict(size=11, color="#475569"), bgcolor="white")

    for n in noeuds:
        fig.add_annotation(
            x=n["x"], y=n["y"], text=n["texte"], showarrow=False,
            font=dict(color="white", size=12),
            bgcolor=n["couleur"], bordercolor="white", borderwidth=2,
            borderpad=6, align="center")

    fig.add_trace(go.Scatter(
        x=[n["x"] for n in noeuds], y=[n["y"] for n in noeuds],
        mode="markers", marker=dict(size=40, opacity=0),
        hovertext=[n["survol"] for n in noeuds], hoverinfo="text"))

    nf = nb_feuilles(root)
    prof = profondeur(root)
    largeur = max(900, 170 * nf)
    hauteur = max(380, 170 * (prof + 1) + 80)

    fig.update_layout(
        title="Arbre de décision",
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(visible=False, range=[-0.7, max(nf - 0.3, 0.7)]),
        yaxis=dict(visible=False, range=[-prof - 0.5, 0.5]),
        width=largeur, height=hauteur,
        margin=dict(l=20, r=20, t=60, b=20))
    return fig
