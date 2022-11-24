import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

<<<<<<< HEAD
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)[0]))
=======
    def forward(self, x, sublayer, if_att=True):
        out = sublayer(self.norm(x))
        if if_att:
            return x + self.dropout(out[0]),  out[1]
        else:
            return x + self.dropout(out)
>>>>>>> 6965938 (update)
