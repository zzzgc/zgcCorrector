# import torch
# from transformers import AutoModel, AutoTokenizer
# from torch import nn
# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# model = AutoModel.from_pretrained("bert-base-chinese").to('cuda')

# i = torch.tensor(tokenizer(['哪里上课的'])['input_ids']).to('cuda')
# print(i.shape)

# e = torch.load('/var/zgcCorrector/data/font/font_embedding.emb').to('cuda')
# glyph_embedding = nn.Embedding(21128,1024).from_pretrained(e).to('cuda')
# l = nn.Linear(1024, 768).to('cuda')
# aa = glyph_embedding(i)
# print('aaa', aa.shape)
# a = model.embeddings(input_ids=i)
# print(a.shape)
# print(model.encoder(a+l(aa))[0].shape)
# print(model.pooler(model.encoder(a).last_hidden_state).shape)


# from transformers import AutoModel, AutoTokenizer
# import torch
# from torch import nn

# path = 'bert-base-chinese'
# model = AutoModel.from_pretrained(path)
# v = torch.tensor([1])
# print(model.embeddings.word_embeddings(v)[0][:10])
# vv = torch.randn([21128, 768])
# model.embeddings.word_embeddings = nn.Embedding(21128,768).from_pretrained(vv)
# print(model.embeddings.word_embeddings(v)[0][:10])



# print(model.embeddings)

# model.embeddings = model.embeddings.word_embeddings + model.embeddings.token_type_embeddings
# print()
# print(model.embeddings)

# output = model.encoder(embeddings)
# print(out)


from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch

discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")

sentence = "The quick brown fox jumps over the lazy dog"
fake_sentence = "The quick brown fox fake over the lazy dog"

fake_tokens = tokenizer.tokenize(fake_sentence, add_special_tokens=True)
fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
discriminator_outputs = discriminator(fake_inputs)
print(discriminator_outputs)


