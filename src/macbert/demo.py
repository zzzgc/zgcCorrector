import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModel.from_pretrained("bert-base-chinese")

i = torch.tensor(tokenizer(['哪里上课的'])['input_ids'])
print(i)
a = model.embeddings(input_ids=i)
print(a)
print(model.encoder(a))
print(model.pooler(model.encoder(a).last_hidden_state))


from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn

path = 'bert-base-chinese'
model = AutoModel.from_pretrained(path)
v = torch.tensor([1])
print(model.embeddings.word_embeddings(v)[0][:10])
vv = torch.randn([21128, 768])
model.embeddings.word_embeddings = nn.Embedding(21128,768).from_pretrained(vv)
print(model.embeddings.word_embeddings(v)[0][:10])



print(model.embeddings)

model.embeddings = model.embeddings.word_embeddings + model.embeddings.token_type_embeddings
print()
print(model.embeddings)

