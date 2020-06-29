import numpy as np
from numpy import dot
from numpy.linalg import norm
import bert_poids_class, bert_class

#########################################################################################
## Here we define a class action which take as parameters :
##      - the story that the journalist is typing
##      - the dictionnary (= client database processed) that we produced with discovery_1.py
## With some few functions we can then make a recommendation
##  See the easy exemple at the end, only few lines are required
## Really fast once the dictionnary is stored (discovery_1.py)
##########################################################################################

def cos(a,b):
    return dot(a, b)/(norm(a)*norm(b))

def story2vec(story):
    texte = story+'.'
    test = bert_poids_class.keywords(texte)
    test.extraction()
    test.remplissage()

    test_bis = bert_class.article(texte)
    test_bis.sent_embeddings()
    test_bis.article_embedding(test.poids)

    return test_bis.article_emb


class action:

    def __init__(self, story, dico):
        self.dico=dico
        self.story=story
        self.storyVec=story2vec(self.story)

    def kw(self):
        aux=bert_poids_class.keywords(self.story)
        aux.extraction()
        self.keywords=[a[0] for a in aux.keyphrases]

    def recommend(self,top=10):
        self.top=top
        self.table=[[k,cos(v[0],self.storyVec),v[1]] for k,v in self.dico.items()]
        self.table.sort(key= lambda x:x[1], reverse=True)
        self.best=[[a[0],[b[0] for b in a[2]]] for a in self.table[:self.top]]



    def affichage(self):
        print('Your story : {}'.format(self.story))
        print("Keywords in your story : {}".format(self.keywords))
        #print("Entities in your story : {}".format(self.entities))
        print("Top {} articles recommendés : ".format(self.top))
        for art in self.best:
            print(" ")
            print("article : {}".format(art[0]))
            print("Keywords of this article : ")
            print(art[1])


if __name__ == '__main__':
#    from sklearn.externals import joblib
    import joblib
    dico=joblib.load("/home/administrator/dan_stage/BERT/dico_586_articles_and_keywords.pkl")
#work on the new article (about 2sec)  --> find the vector b for this article
#dico[url_new_article]=b
#joblib.dump(dico)

    import time
    t1=time.time()
    st="Jazz concert in Chicago"
    test=action(st,dico)
    test.kw()
    test.recommend(6)
    test.affichage()
    print(time.time()-t1, "secondes.")

