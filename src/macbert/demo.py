import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModel.from_pretrained("bert-base-chinese")

i = torch.tensor(tokenizer(['哪里上课的'])['input_ids'])
print(i)
print(model(input_ids=i))