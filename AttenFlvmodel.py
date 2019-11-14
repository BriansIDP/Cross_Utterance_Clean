import torch.nn as nn
from torch import cat, zeros, rand, arange, ger
from torch.autograd import Variable
from SelfAtten import SelfAttenModel

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = ger(pos_seq, self.inv_freq)
        pos_emb = cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]

class AttenFlvModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, nmid,
                 dropout=0.5, tie_weights=False, reset=0, nhead=1):
        """ntoken: vocabulary size
           ninp: word emb size
           nhid: hidden state size
           nlayers: number of RNN layers
           nmid: attention middle layer size
           dropout: RNN dropout rate
           reset: sentence boundary resetting
           nhead: number of attention heads
	"""
        super(AttenFlvModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.selfatten = SelfAttenModel(nhid, nmid, nhead)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.pos_emb = PositionalEmbedding(ninp)
        self.emb_drop = nn.Dropout(dropout)
        self.emb_compressor = nn.Linear(ninp, ninp)
        self.emb_act = nn.ReLU()

        self.init_weights()

        self.nhid = nhid
        self.nhead = nhead
        self.nlayers = nlayers
        self.ninp = ninp
        self.reset = reset
        self.mode = 'train'

    def set_mode(self, m):
        self.mode = m

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, device='cuda', eosidx=1, direct=False):
        """input: word idx
           hidden: initial/carries hidden states
           device: cuda or cpu
           eosidx: end of sequence idx
           direct: use word embeddings directly or after LSTM encoding
        """
        emb = self.drop(self.encoder(input))
        if direct:
            # Adding positional encoding
            pos_seq = arange(input.size(0)-1, -1, -1.0, dtype=emb.dtype).to(device)
            pos_emb = self.pos_emb(pos_seq, input.size(1))
            output = self.emb_compressor(cat([emb, pos_emb], 2))
            output = self.emb_act(output)
            output = self.emb_drop(output)
            output = emb
        else:
            if self.reset:
                output_list = []
                for i in range(emb.size(0)):
                    resethidden = self.resetsent(hidden, input[i,:], eosidx)
                    each_output, hidden = self.rnn(emb[i,:,:].view(1,emb.size(1),-1), resethidden)
                    output_list.append(each_output)
                output = cat(output_list, 0)
            else:
                output, hidden = self.rnn(emb, hidden)
        extracted, penalty = self.selfatten(output.transpose(0,1), device=device, wordlevel=True)
        return extracted, penalty 

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

    def resetsent(self, hidden, input, eosidx):
        outputcell = hidden[0]
        memorycell = hidden[1]
        mask = input != eosidx
        expandedmask = mask.unsqueeze(-1).expand_as(outputcell)
        expandedmask = expandedmask.float()
        return (outputcell*expandedmask, memorycell*expandedmask)
