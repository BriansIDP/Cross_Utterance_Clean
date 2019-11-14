import torch.nn as nn
from torch import cat, zeros
from torch.autograd import Variable
from SelfAtten import SelfAttenModel

class L2RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nutt, nseg, naux, nhid, nlayers, atten=False,
                 dropout=0.5, dropaux=0.5, tie_weights=False, reset=0, nhead=1):
        """rnn_type: choose from LSTM, RNN and GRU
           ntoken: vocabulary size
	   ninp: word embedding dimension
	   nutt: utterance embedding dimension
	   naux: auxiliary feature dimension
	   nhid: hidden state dimension
	   nlayers: number of RNN layers
	   nseg: number of segments
	   atten: whether to use attention for second level LM
	   dropout, dropaux: dropout rates
	   tie_weights: tie input/output weight matrices
	   reset: reset at utterance boundaries
	   nhead: number of attention heads
        """
        super(L2RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        # instantiate attention output layer
        if atten:
            self.selfatten = SelfAttenModel(nutt, nutt, nhead)
            self.comp4atten = nn.Linear(nutt*nhead, naux)
        else:
            self.compressor = nn.Linear(nutt*nseg, naux)
        self.compressDrop = nn.Dropout(dropaux)
        self.compressReLU = nn.ReLU()
        self.compressSig = nn.Sigmoid()
        self.compressTanh = nn.Tanh()
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp+naux, nhid, nlayers, dropout=dropout)
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

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.atten = atten
        self.naux = naux
        self.nhead = nhead
        self.nlayers = nlayers
        self.reset = reset
        self.mode = 'train'
        self.init_weights()

    def set_mode(self, m):
        self.mode = m

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.atten:
            self.comp4atten.bias.data.zero_()
            self.comp4atten.weight.data.uniform_(-initrange, initrange)
        else:
            self.compressor.bias.data.zero_()
            self.compressor.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, auxiliary, hidden, eosidx = 0, target=None, device='cuda'):
        emb = self.drop(self.encoder(input))
        penalty = zeros(1).to(device)
        bsz = auxiliary.size(0)*auxiliary.size(1)
        if self.atten:
            auxiliary_in, penalty = self.selfatten(auxiliary.view(bsz, -1), device=device)
            if self.nutt * self.nhead != self.naux:
                auxiliary_in = self.comp4atten(auxiliary_in)
                auxiliary_in = self.compressDrop(auxiliary_in)
        else:
            auxiliary_in = self.compressor(auxiliary.view(bsz, auxiliary.size(2)))
            auxiliary_in = self.compressDrop(auxiliary_in)
        to_input = cat([auxiliary_in.view(auxiliary.size(0), auxiliary.size(1), -1), emb], 2)
        output_list = []
        if self.reset:
            for i in range(emb.size(0)):
                hidden = self.resetsent(hidden, input[i,:], eosidx)
                each_output, hidden = self.rnn(to_input[i,:,:].view(1,emb.size(1),-1), hidden)
                output_list.append(each_output)
            output = cat(output_list, 0)
        else:
            output, hidden = self.rnn(to_input, hidden)
        output = self.drop(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, penalty

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
