import csv
import bert_class
import bert_poids_class
from scipy.spatial.distance import cosine


def sim(a,b):
    return 1 - cosine(a,b)


data=[]
tsv_file = open("C:\DALET\dataset-sts\data\para\msr\msr-para-test.tsv","r")
read_tsv = csv.reader(tsv_file, delimiter="\t")
i=0
for row in read_tsv:
    print(row)
    if i!=0:
        if len(row)<5:
            a,b=row[3].split('\t')
            data.append([int(row[0]),a,b])
        else:
            data.append([int(row[0]),row[3],row[4]])
    print(i)
    i+=1
    if i>200:
        break # c trop long
tsv_file. close()


sim0, sim1 = 0, 0
nb0, nb1 = 0,0
i=1
for x in data:
    v, a, b = x
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

    if v==0:
        nb0+=1
        sim0+=similarity
    else:
        nb1 += 1
        sim1 += similarity
        #print(sim1)

    print(i, 'sur ', len(data))
    i+=1

print("sim non para : ", sim0/nb0)
print("sim para : ", sim1/nb1)

