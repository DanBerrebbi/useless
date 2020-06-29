import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import logging
import split
import numpy as np
import time
logging.basicConfig(level=logging.INFO)


#######################################################################
######           loading BERT           ###############################
#######################################################################
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
model.eval()


#######################################################################
######           Class to vectorize a sentence            #############
#######################################################################

class sentence:

    def __init__(self,st):
        self.st=st

    def vect_by_sum(self):
        marked_text = "[CLS] " + self.st + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        self.tokenized_text=tokenized_text

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)


        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            #t1=time.time()
            outputs = model(tokens_tensor, segments_tensors)
            #print("temps : {}".format(time.time()-t1))
            hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)


        token_vecs_sum = []
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)
        self.embeddings=token_vecs_sum   # je pourrai faire plusieurs embeddings après, genre un pour la somme ... sel.sum_embedding

    def vect_sentence(self):
        self.vect_by_sum()
        return self.embeddings[1:-1]   # we put out the CLS and SEP tokens


    def print(self):
        for i, token_str in enumerate(self.tokenized_text):
            print(i, token_str)


#######################################################################
######           Class to vectorize an article            #############
#######################################################################


class article:
    def __init__(self,article):
        t1=time.time()
        self.sentences = split.split_into_sentences(article)
        #print("temps de split : {} secondes.".format(time.time()-t1))
        #print("Nombre de phrases : {}".format(len(self.sentences)))

    def sent_embeddings(self):
        embeddings=[]
        for sent in self.sentences:
            aux=sentence(sent)
            embeddings+=aux.vect_sentence()
        self.embeddings=embeddings

    def article_embedding(self,poids=[]):
        if len(poids)==0:
            poids=np.array([1 for k in range(len(self.embeddings))])
        #print(len(poids))
        #print(len(self.embeddings))
        assert len(poids)==len(self.embeddings)  #souvent qd ya un pb ici c que il manque un point en fin de phrase ou une betise comme ca.
        self.article_emb=torch.zeros(768)
        for k in range(len(poids)):
            self.article_emb+=poids[k]*self.embeddings[k]
        return self.article_emb