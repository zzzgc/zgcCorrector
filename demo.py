import torch
import torch.nn as nn

def getans(v1, v2):
    v1 = torch.tensor(v1, dtype=float)
    v2 = torch.tensor(v2, dtype=float)
    
    ans = torch.sum(torch.mul(v1, v2)) / (torch.sum(v1*v1) * torch.sum(v2*v2)) ** (1/2)
    
    cos = nn.CosineSimilarity(dim=-1)
    ans_gt = cos(v1, v2)
    
    return ans, ans_gt

def getCosineSimilarity(m1, m2):
    m1 = torch.tensor(m1, dtype=float)
    m2 = torch.tensor(m2, dtype=float)
    
    ans = torch.sum(m1*m2, axis=-1) / (torch.sum(m1*m1, axis=-1) * torch.sum(m2*m2, axis=-1)) ** (1/2)
    
    return ans

v1 = [[1, 2, 3], [4, 5, 6]]
v2 = [[2, 4, 6], [8, 10, 12]]

print(getCosineSimilarity(v1, v2))