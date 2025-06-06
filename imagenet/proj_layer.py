import numpy as np
import torch
import torch.nn as nn


class ProjLayer(nn.Module):
    def __init__(self, inputsize):
        super(ProjLayer, self).__init__()
        
        self.d = self.tensor_prompt(inputsize, inputsize//32, init='he_uniform')
        self.d_2 = self.tensor_prompt(inputsize//32, inputsize//128, init='he_uniform')
        self.d_3 = self.tensor_prompt(inputsize//128, 4, init='he_uniform')
        self.gelu = nn.GELU()
       
    def forward(self, x):
        x = torch.einsum('ji,ib->jb', x, self.d)
        x = self.gelu(x)
        x = torch.einsum('jb,bk->jk', x, self.d_2)
        x = self.gelu(x)
        x = torch.einsum('jk,kl->jl', x, self.d_3)

        return x

    def tensor_prompt(self, x, y=None, z=None, w=None, init='he_uniform', gain=1, std=1, a=1, b=1):
        if y is None:
            p = torch.nn.Parameter(torch.FloatTensor(x), requires_grad=True)
        elif z is None:
            p = torch.nn.Parameter(torch.FloatTensor(x,y), requires_grad=True)
        elif w is None:
            p = torch.nn.Parameter(torch.FloatTensor(x,y,z), requires_grad=True)
        else:
            p = torch.nn.Parameter(torch.FloatTensor(x,y,z,w), requires_grad=True)

        if p.dim() > 2:
            self.tensor_init(p[0], init, gain=gain, std=std, a=a, b=b)
            for i in range(1, x): p.data[i] = p.data[0]
        else:
            self.tensor_init(p, init)
        
        return p

    def tensor_init(self, p, init, gain=1, std=1, a=1, b=1):
        if init == 'ortho':
            nn.init.orthogonal_(p)
        elif init == 'uniform':
            nn.init.uniform_(p, a=a, b=b)
        elif init == 'normal':
            nn.init.normal_(p, std=std)
        elif init == 'zero':
            nn.init.zeros_(p)
        elif init == 'he_uniform':
            nn.init.kaiming_uniform_(p, a=a)
        elif init == 'he_normal':
            nn.init.kaiming_normal_(p, a=a)
        elif init == 'xavier_uniform':
            nn.init.xavier_uniform_(p, gain=gain)
        elif init == 'xavier_normal':
            nn.init.xavier_normal_(p, gain=gain)
        elif init == 'trunc_normal':
            nn.init.trunc_normal_(p, std=std)
        else:
            assert NotImplementedError


class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc(x.squeeze(1))) 
        return x