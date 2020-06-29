# le but de ce fichier est de faire un matching au niveau des poids
# Yake et positionRank ne sont pas ds le mm ordre, pr Yake petit c bien et c le contraire pr PositionRank
import pke
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def normalize(v):   #normaliser en norme 2
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

class keywords:

    def __init__(self,texte):
        if type(texte)!='str':
            self.texte=str(texte)
        else :
            self.texte=texte
        # création d'un dico avec les tokens.
        self.tokenized_text = tokenizer.tokenize(texte)
        self.dico_poids = {x: 0 for x in self.tokenized_text}

    def extraction(self):
        extractor = pke.unsupervised.PositionRank()
        extractor.load_document(self.texte, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        if len(extractor.candidates)>=50:
            self.keyphrases = extractor.get_n_best(n=len(extractor.candidates) // 2)
        elif len(extractor.candidates)<2:
            aux=self.texte.split(' ')
            aux2=[[x,.5] for x in aux]
            self.keyphrases=aux2
        else :
            self.keyphrases = extractor.get_n_best(n=len(extractor.candidates))
        #print(self.keyphrases)

    def remplissage(self):
        for x in self.keyphrases:
            phrase, score = x
            mots = tokenizer.tokenize(phrase)
            for mot in mots:
                if mot in self.dico_poids.keys():
                    self.dico_poids[mot] += score * 5
        self.poids = []
        for x in self.tokenized_text:
            self.poids.append(self.dico_poids[x])
        self.poids = np.array(self.poids)
        self.poids = normalize(self.poids)
