import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import autograd

class MaxDropout(nn.Module):
    def __init__(self, drop=0.3):
#         print(p)
        super(MaxDropout, self).__init__()
        if drop < 0 or drop > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.drop = 1 - drop

    def forward(self, x):
        if not self.training:
            return x

        up = x - x.min()
        divisor =  (x.max() - x.min())
        x_copy = torch.div(up,divisor)
        if x.is_cuda:
            x_copy = x_copy.cuda()

        mask = (x_copy > (self.drop))
        x = x.masked_fill(mask > 0.5, 0)
        return x 


class AlphaDropout(nn.Module):
    # Custom implementation of alpha dropout. Note that an equivalent
    # implementation exists in pytorch as nn.AlphaDropout
    def __init__(self, dropout=0.1, lambd=1.0507, alpha=1.67326):
        super().__init__()
        self.lambd = lambd
        self.alpha = alpha
        self.aprime = -lambd * alpha
        
        self.q = 1 - dropout
        self.p = dropout

        self.a = (self.q + self.aprime**2 * self.q * self.p)**(-0.5)
        self.b = -self.a * (self.p * self.aprime)
        
    def forward(self, x):
        if not self.training:
            return x
        ones = torch.ones(x.size())
        x_copy = (x - x.min()) / (x.max() - x.min()).detach().clone()
        if x.is_cuda:
            ones = ones.cuda()
            x_copy = x_copy.cuda()
        mask = (x_copy > (self.q))
        x = x.masked_fill(autograd.Variable(mask.bool()), 0)
        return x 

