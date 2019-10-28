# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from operator import itemgetter

import dataloader
import model

arglist = []
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/AMI',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to embeddings (0 = no dropout)')
parser.add_argument('--rnndrop', type=float, default=0.2,
                    help='dropout applied to rnns (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--evalmode', action='store_true',
                    help='Evaluation only mode')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--reset', type=int, default=0,
                    help='reset on the sentence boundaries')
parser.add_argument('--loss', type=str, default='ce',
                    help='loss functions to be used')
parser.add_argument('--noise_ratio', type=int, default=50,
                    help='set the noise ratio of NCE sampling, the noise')
parser.add_argument('--norm_term', type=int, default=9,
                    help='set the log normalization term of NCE sampling')
parser.add_argument('--factor', type=float, default=0.5,
                    help='interpolation value')
parser.add_argument('--interp', action='store_true',
                    help='Linear interpolate with Ngram')
parser.add_argument('--stream_out', action='store_true',
                    help='Write out stream')
parser.add_argument('--logfile', type=str, default='LOGs/rnn.log',
                    help='path to save the final model')
args = parser.parse_args()

arglist.append(('Data', args.data))
arglist.append(('Model', args.model))
arglist.append(('Embedding Size', args.emsize))
arglist.append(('Hidden Layer Size', args.nhid))
arglist.append(('Layer Number', args.nlayers))
arglist.append(('Learning Rate', args.lr))
arglist.append(('Update Clip', args.clip))
arglist.append(('Max Epochs', args.epochs))
arglist.append(('BatchSize', args.batch_size))
arglist.append(('Sequence Length', args.bptt))
arglist.append(('Dropout', args.dropout))
arglist.append(('Loss Function', args.loss))
arglist.append(('Noise Ration', args.noise_ratio))
arglist.append(('Norm Term', args.norm_term))

def logging(s, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(args.logfile, 'a+') as f_log:
            f_log.write(s + '\n')

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        logging("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################
dictionary = dataloader.Dictionary(os.path.join(args.data, 'dictionary.txt'))
eosidx = dictionary.get_eos()
# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
if args.evalmode:
    eval_batch_size = args.eval_batch_size

# Stream writeout mode
if args.stream_out and eval_batch_size != 1:
    logging('Batch size must be 1 in stream writeout mode!')
    raise

# Export vocabulary
# with open(os.path.join(args.data, 'dictionary.txt'), 'w') as vocabout:
#     for ind, word in enumerate(corpus.dictionary.idx2word):
#         vocabout.write(str(ind)+' '+word+'\n')

###############################################################################
# Build the model
###############################################################################
ntokens = len(dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.rnndrop, args.dropout, args.tied, reset=args.reset)
criterion = nn.CrossEntropyLoss()
interpCrit = nn.CrossEntropyLoss(reduction='none')

if args.cuda:
    model.cuda()
###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source, ngramProb=None):
    # Turn on evaluation mode which disables dropout.
    model.to(device)
    model.eval()
    model.set_mode('eval')
    total_loss = 0.
    stout = []
    ntokens = len(dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            # import pdb; pdb.set_trace()
            if ngramProb != None:
                _, batch_ngramProb = get_batch(ngramProb, i)
            # gs534 add sentence resetting
            eosidx = dictionary.get_eos()
            output, hidden = model(data, hidden, separate=args.reset, eosidx=eosidx)
            output_flat = output.view(-1, ntokens)
            logProb = interpCrit(output.view(-1, ntokens), targets)
            rnnProbs = torch.exp(-logProb)
            if args.interp and args.evalmode:
                final_prob = args.factor * rnnProbs + (1 - args.factor) * batch_ngramProb
            else:
                final_prob = rnnProbs
            if args.stream_out:
                stout += final_prob.tolist()
            total_loss += (-torch.log(final_prob).sum()) / data.size(1)
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source), stout

def train(model, train_data, lr):
    # Turn on training mode which enables dropout.
    model.train()
    model.set_mode('train')
    total_loss = 0.
    start_time = time.time()
    ntokens = len(dictionary)
    hidden = model.init_hidden(args.batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.wdecay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        # gs534 add sentence resetting
        eosidx = dictionary.get_eos()
        if args.loss == 'nce':
            output, hidden = model(data, hidden, eosidx, targets)
            loss = criterion(output)
            loss.backward()
        else:
            output, hidden = model(data, hidden, separate=args.reset, eosidx=eosidx)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        #for p in model.parameters():
        #    p.data.add_(-lr, p.grad.data)
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):
    logging('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)

def loadNgram(path):
    probs = []
    with open(path) as fin:
        for line in fin:
            probs.append(float(line.strip()))
    return torch.Tensor(probs)

logging('Training Start!')
for pairs in arglist:
    logging(pairs[0] + ': ' + str(pairs[1]))
# Loop over epochs.
lr = args.lr
best_val_loss = None
train_loader, val_loader, test_loader = dataloader.create(args.data, batchSize=1, workers=0)

# At any point you can hit Ctrl + C to break out of training early.
if not args.evalmode:
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            for train_batched in train_loader:
                train_data = batchify(train_batched, args.batch_size)
                train(model, train_data, lr)
            aggregate_valloss = 0.
            total_valset = 0
            for val_batched in val_loader:
                databatchsize = val_batched.size()[0]
                val_data = batchify(val_batched, eval_batch_size)
                val_loss, _ = evaluate(val_data)
                aggregate_valloss = aggregate_valloss + databatchsize * val_loss
                total_valset += databatchsize
            val_loss = aggregate_valloss / total_valset    
            logging('-' * 89)
            logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                val_loss, math.exp(val_loss)))
            logging('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 2.0
    except KeyboardInterrupt:
        logging('-' * 89)
        logging('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Set cpu evaluate mode
device = torch.device("cuda")

# Prepare n_gram for evaluation
if args.interp:
    TestNgramData = loadNgram(os.path.join(args.data, 'test_ngram.st'))
    TestNgramProbs = batchify(TestNgramData, eval_batch_size)
    ValNgramData = loadNgram(os.path.join(args.data, 'valid_ngram.st'))
    ValNgramProbs = batchify(ValNgramData, eval_batch_size)
else:
    TestNgramProbs = None
    ValNgramProbs = None

# Run on test data.
test_start_time = time.time()
# Write out probabilities
if args.stream_out and args.evalmode:
    devstfile = open(args.data+'dev.st', 'w')
    evalstfile = open(args.data+'eval.st', 'w')
total_testset = 0
aggregate_testloss = 0.
for test_batched in test_loader:
    databatchsize = test_batched.size(0)
    test_data = batchify(test_batched, eval_batch_size)
    test_loss, stout = evaluate(test_data.to(device), TestNgramProbs)
    if args.stream_out:
        evalstfile.writelines([str(f)+'\n' for f in stout])
    aggregate_testloss = aggregate_testloss + databatchsize * test_loss
    total_testset += databatchsize
test_loss = aggregate_testloss / total_testset
logging('=' * 89)
logging('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logging('=' * 89)
logging('Test time cost: {:5.2f} ms'.format(time.time() - test_start_time))

# Run on valid data.
if args.evalmode:
    total_valset = 0
    aggregate_valloss = 0.
    for val_batched in val_loader:
        databatchsize = val_batched.size(0)
        val_data = batchify(val_batched, eval_batch_size)
        val_loss, stout = evaluate(val_data.to(device), ngramProb=ValNgramProbs)
        if args.stream_out:
            devstfile.writelines([str(f)+'\n' for f in stout])
        aggregate_valloss = aggregate_valloss + databatchsize * val_loss
        total_valset += databatchsize
    val_loss = aggregate_valloss / total_valset
    logging('=' * 89)
    logging('| End of training | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
        val_loss, math.exp(val_loss)))
    logging('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)

