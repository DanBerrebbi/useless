import bert_poids_class, bert_class
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
import time
import argparse

def cos(a,b):
    return dot(a, b)/(norm(a)*norm(b))

if __name__ == "__main__":

    #texte=open("C:\DALET\\nytimes_news_articles.txt","r",encoding="utf8").read()
    texte=open("/home/administrator/dan_stage/nytimes_news_articles.txt","r",encoding="utf8").read()
    split=texte.rsplit('\n\n')
    t1=time.time()
    dico_art={}
    for k in range(0,len(split),2):
       dico_art[split[k]]=split[k+1]+'.'   #tjrs le pb des phrases où il manque des points.

    # je garde que 20 articles pour l'instant
    small_dico={}
    for k,v in dico_art.items():
        if len(small_dico)<6000:
            small_dico[k]=v


    dico_vect={}
    i=1
    for k,v in small_dico.items():
        try:            # try and except because some articles of this dataset can't be read (about 0.3 % so it's okay for now)
            texte=v
            test = bert_poids_class.keywords(texte)
            test.extraction()
            kw=test.keyphrases
            test.remplissage()

            test_bis = bert_class.article(texte)
            test_bis.sent_embeddings()
            test_bis.article_embedding(test.poids)

            dico_vect[k]=(test_bis.article_emb,kw)
            print(i, "temps : {} secondes.".format(time.time()-t1))
            i+=1
        except:
            print("L'article {} n'a pas pu etre lu".format(k))


    #from sklearn.externals import joblib
    import joblib
    #output_file="C:\DALET\BERT\\dico_6000_articles_and_keywords.pkl"
    output_file="/home/administrator/dan_stage/BERT/dico_6000_articles_and_keywords.pkl"
    joblib.dump(dico_vect,output_file)


