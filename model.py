from __future__ import print_function
import torch.nn as nn
from torch import cat
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, rnndrop=0.5, dropout=0.5, tie_weights=False, reset=0):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=rnndrop)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.reset = reset
        self.mode = 'train'
        self.grads = {}

    def set_mode(self, m):
        self.mode = m

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, separate=0, eosidx = 0, target=None, outputflag=0, hiddenpos=0):
        emb = self.drop(self.encoder(input))
        output_list = []
        return_hidden = hidden
        if separate == 1:
            for i in range(emb.size(0)):
                resethidden = self.resetsent(hidden, input[i,:], eosidx)
                each_output, hidden = self.rnn(emb[i,:,:].view(1,emb.size(1),-1), resethidden)
                output_list.append(each_output)
            output = cat(output_list, 0)
        elif separate == 2:
            for i in range(emb.size(0)):
                each_output, hidden = self.rnn(emb[i,:,:].view(1,emb.size(1),-1), hidden)
                output_list.append(each_output)
                if i == hiddenpos:
                    return_hidden = hidden
            output = cat(output_list, 0)
        else:
            output, hidden = self.rnn(emb, hidden)

        output = self.drop(output)

        if separate == 2:
            hidden = return_hidden

        if outputflag == 0:
            decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
        else:
            return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def resetsent(self, hidden, input, eosidx):
        if self.rnn_type == 'LSTM':
            outputcell = hidden[0]
            memorycell = hidden[1]
            mask = input != eosidx
            expandedmask = mask.unsqueeze(-1).expand_as(outputcell)
            expandedmask = expandedmask.float()
            return (outputcell*expandedmask, memorycell*expandedmask)
        else:
            mask = input != eosidx
            expandedmask = mask.unsqueeze(-1).expand_as(hidden)
            expandedmask = expandedmask.float()
            return hidden*expandedmask
