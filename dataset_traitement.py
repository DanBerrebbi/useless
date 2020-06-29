import numpy as np

class dataset:

    def __init__(self,fichier):
        self.txt=fichier

    def convert_to_tab(self):
        data = []
        l = []
        with open(self.txt, "r") as f:
            l = f.readlines()

        for i in l[1:]:
            aux = i.split('\t')
            data.append([aux[1]+'.', aux[2]+'.', float(aux[3])])  #ATTENTION IL FAUT RAJOUTER DES POINTS A LA FIN DES PHRASES SINON CA BEUG

        data = np.array(data)
        return data

