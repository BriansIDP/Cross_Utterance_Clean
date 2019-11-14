import torch.nn as nn
from torch import cat, rand, zeros, matmul, eye, set_printoptions
from torch.autograd import Variable

class SelfAttenModel(nn.Module):
    """Implementation of self-attentive layer"""
    def __init__(self, ninp, ninterm, nweights, dropinter=0.2, dropatt=0):
        """ninp: input hidden state dimension
           ninterm: intermediate dimension
           nweights: number of attention weights
           dropinter: intermediate stage dropout rate
           dropatt: attention dropout rate
        """
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
        
    def forward(self, embs, scale=1, device='cuda', wordlevel=False):
        """embs: input embeddings, could be word or segment embedding
	   scale: scale of the penalty term
	   device: cuda or cpu
	   wordlevel: what level are the embeddings at
	"""
        if not wordlevel:
            if embs.size(1) % self.ninp != 0:
                print('Splitting of input embedding is invalid!')
                raise
            embs = embs.view(embs.size(0), -1, self.ninp)
        intermediate = self.tanh(self.layer1(embs))
        annotmatrix = self.softmax(self.layer2(intermediate))
        totaloutput = matmul(annotmatrix.transpose(1,2), embs)
        ATA = matmul(annotmatrix.transpose(1,2), annotmatrix)
        I = eye(self.nweights).to(device)
        penalty = scale * ((ATA - I.expand_as(ATA)) ** 2).sum()
        return totaloutput.view(embs.size(0), -1), penalty

if __name__ == "__main__":
    a = rand(3,2,5)
    testatten = SelfAttenModel(5, 5, 2)
    output, penalty = testatten(a, wordlevel=True)
