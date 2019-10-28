import torch.nn as nn
from torch import cat, rand, zeros, matmul, eye, set_printoptions
from torch.autograd import Variable

class SelfAttenModel(nn.Module):
    '''Implementation of self-attentive layer'''
    def __init__(self, ninp, ninterm, nweights, dropinter=0.2, dropatt=0):
        super(SelfAttenModel, self).__init__()
        self.layer1 = nn.Linear(ninp, ninterm*nweights, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.layer2 = nn.Linear(ninterm*nweights, nweights, bias=False)
        self.ninp = ninp
        self.nweights = nweights
        self.ninterm = ninterm*nweights
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.layer1.weight.data.uniform_(-initrange, initrange)
        self.layer2.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, uttemb, scale=1, device='cuda', writeout=False, wordlevel=False):
        if not wordlevel:
            if uttemb.size(1) % self.ninp != 0:
                print('Splitting of input embedding is invalid!')
                raise
            uttemb = uttemb.view(uttemb.size(0), -1, self.ninp)
        intermediate = self.tanh(self.layer1(uttemb))
        annotmatrix = self.softmax(self.layer2(intermediate))
        totaloutput = matmul(annotmatrix.transpose(1,2), uttemb)
        ATA = matmul(annotmatrix.transpose(1,2), annotmatrix)
        I = eye(self.nweights).to(device)
        penalty = scale * ((ATA - I.expand_as(ATA)) ** 2).sum()
        return totaloutput.view(uttemb.size(0), -1), penalty

if __name__ == "__main__":
    a = rand(3,2,5)
    testatten = SelfAttenModel(5, 5, 2)
    output, penalty = testatten(a, wordlevel=True)
