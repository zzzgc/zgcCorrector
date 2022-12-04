from os import truncate
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def klloss():
    a = torch.tensor([[[1,0,1.1,2,5,0,0],[1,1,1,2,1,0,0]], [[1,0,1.1,2,5,0,0],[1,1,1,2,1,0,0]]])

    b = torch.tensor([[[1,0,1.1,2,5,0,0],[1,1,1,2,1,0,0]], [[1,0,1.1,2,5,0,0],[1,1,1,2,1,0,0]]])
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    # input should be a distribution in the log space
    logSoftMax = nn.LogSoftmax(dim=-1)
    softMax = nn.Softmax(dim=-1)
    input = logSoftMax(a)
    # Sample a batch of distributions. Usually this would come from the dataset
    target = softMax(b)
    output = kl_loss(input, target)
    print(output)
    
    a = torch.tensor([[[1,0,1.1,2,5,0,0],[1,1,1,2,1,0,0]],[[1,0,1.1,2,5,0,0],[1,1,1,2,1,0,0]]])
    b = torch.tensor([[1,0,1.1,2,5,0,0],[1,1,1,2,1,0,0]])
    print(a.size(), b.size())
    
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    # input should be a distribution in the log space
    logSoftMax = nn.LogSoftmax(dim=-1)
    softMax = nn.Softmax(dim=-1)
    input = logSoftMax(a)
    # Sample a batch of distributions. Usually this would come from the dataset
    target = softMax(b)
    l = 0
    output = kl_loss(input, target)
    print(output)
    l += output
    print(l)

def infonce():
    b = torch.tensor([[[1,0,1.1,2,5,0,0],[1,1,1,2,1,0,0],[1,0,1.1,2,5,0,0],[1,1,1,2,1,0,0]],[[1,0,1.1,2,5,0,0],[1,1,1,2,1,0,0],[1,0,1.1,2,5,0,0],[1,1,1,2,1,0,0]]])
    
    a = torch.tensor([[[1,1,1.1,2,5,1,0],[1,1,1,2,1,0,0],[1,0,1.1,2,5,9,0],[9,9,9,2,1,0,0]], [[1,0,1.1,2,5,0,9],[9,9,1,9,1,0,0],[1,0,9.1,2,5,0,0],[1,9,1,2,1,0,0]]])
    
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    print(a.size())
    print(38,a)
    shape = a.shape
    a = a.view(-1, shape[-1])
    b = b.view(-1, shape[-1])
    y_true = torch.arange(a.shape[0])
    print(40,y_true)
    print(42,y_true)

    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    print(a.size(), a.unsqueeze(1).size(), b.unsqueeze(0).size())
    sim = F.cosine_similarity(a.unsqueeze(1), a.unsqueeze(0), dim=-1)
    print(46,sim)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    print('**********')
    print(torch.eye(a.shape[0]))
    sim = sim - torch.eye(a.shape[0]) * 1e12
    print(49,sim)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    print(49,sim)
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    print(55,loss)
    # print(klloss())
    
def bertdemo():
    # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    # transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    src = torch.rand(1,2)
    # out = transformer_encoder(src)
    # print(out.shape)
    print(src)
    print(1-src)
    
def tranformer():
    m = nn.TransformerEncoderLayer(d_model=768, nhead=16)
    print(m)
    
def fun():
    x1 = torch.as_tensor([[2., 1., 1.], [2., 1., 1.]])
    x2 = torch.as_tensor([[0., 1., 1.], [0., 1., 1.]])
    x3 = torch.as_tensor([[float("-inf"), 1., 1.], [float("-inf"), 1., 1.]])
    y = torch.as_tensor([0, 1])
    l1 = torch.nn.CrossEntropyLoss(reduction="mean")
    l2 = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
    # l3 = torch.nn.CrossEntropyLoss(weight=torch.as_tensor([0., 1., 1.]), reduction="none")
    print(l1(x1, y))
    print(l2(x1, y))
from transformers import ElectraForPreTraining
class DetectModel(nn.Module):
    def __init__(self, DetectModelPath):
        super(DetectModel, self).__init__()
        self.bert = ElectraForPreTraining.from_pretrained(DetectModelPath)
        self.type = type
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, **kwarg):
        output = self.bert(input_ids, token_type_ids, attention_mask, labels)
        loss = None
        output = output.logits
        loss = output.loss
        # output:[batch, seq_len]
        return loss, output

def electra():
    
    from pypinyin import lazy_pinyin, Style
    print(lazy_pinyin('四月是你的谎言,，绿色',style = Style.TONE3))
    print(lazy_pinyin('四月是你的谎言',style = Style.INITIALS))
    print(lazy_pinyin('四月是你的谎言',style = Style.FIRST_LETTER ))
    print(lazy_pinyin('四月是你的谎言',style = Style.FINALS  ))
electra()