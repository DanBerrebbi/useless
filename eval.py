import dataset_traitement
import bert_poids_class
import bert_class
from scipy.spatial.distance import cosine

#fonctions utiles :

def sim(a,b):
    return 1 - cosine(a,b)

def intersection(lst1, lst2):
    return [item for item in lst1 if item in lst2]

fichier = "C:\DALET\dataset-sts\data\sts\sick2014\SICK_trial.txt"

class eval :

    def __init__(self,fichier,nb=0):
        dataset = dataset_traitement.dataset(fichier)
        data = dataset.convert_to_tab()
        if nb == 0:
            self.nb = len(data)
        else :
            self.nb = nb
        self.data = data[:self.nb]
        self.results=[]   # liste  id, score, similarity

    def calculs(self):
        i = 1
        for x in self.data:
            a, b, s = x
            s = float(s) / 5
            test = bert_poids_class.keywords(a)
            test.extraction()
            test.remplissage()
            test_bis = bert_class.article(a)
            test_bis.sent_embeddings()
            test_bis.article_embedding(test.poids)

            test2 = bert_poids_class.keywords(b)
            test2.extraction()
            test2.remplissage()
            test2_bis = bert_class.article(b)
            test2_bis.sent_embeddings()
            test2_bis.article_embedding(test2.poids)

            similarity = sim(test_bis.article_emb, test2_bis.article_emb)
            self.results.append([i, s, similarity])
            print("Etape {} sur {}".format(i, self.nb))
            i += 1

    def performances(self, seuil=.8):
        self.E1, self.E2 = 0, 0  # EAM, EQM
        self.seuil = seuil  # confusion
        self.TP, self.FP, self.FN, self.TN = 0, 0, 0, 0

        for res in self.results:
            s, similarity = res[1:]
            if s > self.seuil:
                if similarity > self.seuil:
                    self.TP += 1
                else:
                    self.FN += 1
            else:
                if similarity <= self.seuil:
                    self.TN += 1
                else:
                    self.FP +=1

        self.precision = self.TP / (self.TP + self.FP)
        self.recall = self.TP / (self.TP + self.FN)
        self.f1 = (self.precision * self.recall * 2) / (self.precision + self.recall)


    def performances_classement(self,top=20):
        assert top!=0
        self.top=top
        tab_score=[x for x in self.results]
        tab_sim=[x for x in self.results]
        tab_score.sort(key= lambda x:x[1])
        tab_sim.sort(key= lambda x:x[2])
        tab_score_top=[x[0] for x in tab_score[:top]]
        tab_sim_top=[x[0] for x in tab_sim[:top]]
        inter=intersection(tab_score_top,tab_sim_top)
        self.classements=len(inter)/self.top


    def affichage(self):
        print("Erreur absolue moyenne : ", self.E1/self.nb)
        print("Erreur quadratique moyenne : ", self.E2/self.nb)
        print("Matrice de confusion pour le seuil ", self.seuil)
        print([[self.TP,self.FN],[self.FP, self.TN]])
        print()
        print("precision : ", self.precision)
        print("recall : ", self.recall)
        print("f1-score : ", self.f1)
        print()
        print("Classements : top {}, valeur : {}".format(self.top, self.classements))


