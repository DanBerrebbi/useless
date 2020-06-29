import torch
from transformers import BertTokenizer, BertModel

import logging
logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
model.eval()


text = """
In the early morning hours of Labor Day last year, a group of gunmen from the 8-Trey street gang made their way through a crowd of revelers gathered near a Brooklyn public housing project to celebrate J’ouvert, a pre-dawn party that precedes the annual West Indian American Day Parade. The housing project was “enemy territory,” the authorities said, the stronghold of a rival gang, the Folk Nation.
As hundreds of people drank and danced in costume, the warring factions spotted each other, and a gunfight broke out in the darkness. Caught in the crossfire was an up-and-coming lawyer in the administration of Gov. Andrew M. Cuomo, Carey Gabay, who was at the festivities with his brother. Mr. Gabay, 43, who was of Jamaican heritage, died a week later from his wounds.
For some in government and law enforcement circles, the death of Mr. Gabay, who had risen from a childhood in Bronx public housing to Harvard Law School and then to public service, was emblematic of the havoc that street gangs have inflicted on New York City residents.
On Wednesday, as part of a continuing investigation, the authorities announced that three men had been indicted in the killing."""


marked_text = "[CLS] " + text + " [SEP]"
tokenized_text = tokenizer.tokenize(marked_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [1] * len(tokenized_text)

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]

token_embeddings = torch.stack(hidden_states, dim=0)
token_embeddings = torch.squeeze(token_embeddings, dim=1)
token_embeddings = token_embeddings.permute(1,0,2)

token_vecs_sum = []
for token in token_embeddings:
    sum_vec = torch.sum(token[-4:], dim=0)
    token_vecs_sum.append(sum_vec)


for i, token_str in enumerate(tokenized_text):
  print (i, token_str)

from scipy.spatial.distance import cosine

diff_bank = 1 - cosine(token_vecs_sum[163], token_vecs_sum[50])
same_bank = 1 - cosine(token_vecs_sum[0], token_vecs_sum[0])

print('Vector similarity for  *similar*  meanings:  %.2f' % same_bank)
print('Vector similarity for *different* meanings:  %.2f' % diff_bank)