class Noeud:
    def __init__(self, n, impurete, prediction, repartition=None):
        self.n = n
        self.impurete = impurete
        self.prediction = prediction
        self.repartition = repartition   
        self.attribut = None
        self.seuil = None
        self.gauche = None
        self.droit = None

    def est_feuille(self):
        return self.droit is None and self.gauche is None
