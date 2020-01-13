# coding: utf-8
import time
import argparse
import sys, os
import torch
import torch.nn as nn
import math
import gc

import L2joint_dataloader_atten
from model import RNNModel
from L2model_debug import L2RNNModel
from AttenFlvmodel import AttenFlvModel

arglist = []
parser = argparse.ArgumentParser(description='PyTorch Level-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/AMI',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=256,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=256,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--FLlr', type=float, default=10,
                    help='initial learning rate')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--FLvclip', type=float, default=0.5,
                    help='first level LM gradient clipping')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--naux', type=int, default=128,
                    help='auxiliary context info feature dimension (after compressor)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--save', type=str, default='model.pt',
                    help='location of the model to be saved')
parser.add_argument('--FLvsave', type=str, default='FLvmodel.pt',
                    help='location of the FLv model to be saved')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--reset', type=int, default=0,
                    help='reset at sentence boundaries')
parser.add_argument('--batchsize', type=int, default=32,
                    help='Batch size used for training')
parser.add_argument('--bptt', type=int, default=35,
                    help='bptt steps used for training')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--seed', type=int, default=1000,
                    help='random seed')
parser.add_argument('--useatten', action='store_true',
                    help='Use self attentive mechanism')
parser.add_argument('--context', type=str, default='0',
                    help='Number of context used')
parser.add_argument('--nhead', type=int, default=1,
                    help='Head number for multi-head self-attention')
parser.add_argument('--alpha', type=float, default=0.01,
                    help='Penalty term scale for multi-head self-attention')
parser.add_argument('--evalmode', action='store_true',
                    help='Evaluation only mode')
parser.add_argument('--factor', type=float, default=0.5,
                    help='interpolation value')
parser.add_argument('--interp', action='store_true',
                    help='Linear interpolate with Ngram')
parser.add_argument('--useletter', action='store_true',
                    help='Use letter ngram in word embeddings')
parser.add_argument('--FLvmodel', type=str, default='model.pt',
                    help='location of the first level model')
parser.add_argument('--updatedelay', type=int, default=1,
                    help='Accumulate gradients for FLvmodel')
parser.add_argument('--outputcell', type=int, default=0,
                    help='How many output cells to be used')
parser.add_argument('--logfile', type=str, default='trainlog.txt',
                    help='location of the logfile')
parser.add_argument('--scratch', action='store_true',
                    help='train First level LM from scratch')
parser.add_argument('--directemb', action='store_true',
                    help='First level LM without RNN but add positional embeddings')
parser.add_argument('--maxlen_prev', type=int, default=30,
                    help='Maximum number of words to look back')
parser.add_argument('--maxlen_post', type=int, default=30,
                    help='Maximum number of words to look back')
parser.add_argument('--use_sampling', action='store_true',
                    help='Use error sampling')
parser.add_argument('--errorfile', type=str, default='confusion.txt',
                    help='location of the confusion pair file')
parser.add_argument('--reference', type=str, default='train.ref',
                    help='location of the reference file')
parser.add_argument('--ratio', type=float, default=1,
                    help='error sampling ratio')
parser.add_argument('--sample_freq', type=int, default=30,
                    help='sample every x epochs')
parser.add_argument('--randsample', action='store_true',
                    help='sample randomly, no acoustic error distributions')
args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu")

def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(args.logfile, 'a+') as f_log:
            f_log.write(s + '\n')

arglist.append(('Data', args.data))
arglist.append(('Model', args.model))
arglist.append(('Embedding Size', args.emsize))
arglist.append(('Auxiliary Input Size', args.naux))
arglist.append(('Hidden Layer Size', args.nhid))
arglist.append(('Layer Number', args.nlayers))
arglist.append(('Learning Rate', args.lr))
arglist.append(('Update Clip', args.clip))
arglist.append(('Max Epochs', args.epochs))
arglist.append(('BatchSize', args.batchsize))
arglist.append(('Sequence Length', args.bptt))
arglist.append(('Dropout', args.dropout))
arglist.append(('Weight Decay', args.wdecay))
arglist.append(('Context', args.context))
arglist.append(('Update delay', args.updatedelay))
arglist.append(('No. of output cells', args.outputcell))
arglist.append(('First level LM', args.FLvmodel))
arglist.append(('Train from scratch', args.scratch))
arglist.append(('Max no. of previous words', args.maxlen_prev))
arglist.append(('Max no. of future words', args.maxlen_post))

if args.useatten:
    logging('Using multi-head self-attention with head number: ')
    logging(str(args.nhead))

# define seed, learning rate and context to be used in the following code
torch.manual_seed(args.seed)
lr = args.lr
FLlr = args.FLlr
eval_batch_size = 10
context = args.context.strip().split(' ')
context = [int(i) for i in context]
embmultsize = args.outputcell
if args.outputcell == 0:
    embmultsize = 1

# Tools
def showmem():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except: pass

def get_needed_utterance_old(utt_index, utt_prev, utt_post, bsz, bptt):
    prev_context = torch.index_select(utt_prev, 0, utt_index)
    post_context = torch.index_select(utt_post, 0, utt_index)
    return prev_context.view(bptt, bsz, -1), post_context.view(bptt, bsz, -1)

def get_needed_utterance(utt_index, utt_prev, utt_post):
    seen_utt = {}
    prev_sent_needed = []
    post_sent_needed = []
    virtual_address = []
    virtual_pointer = 0
    for utt_id in utt_index:
        if utt_id.item() not in seen_utt:
            prev_sent_needed.append(utt_prev[utt_id])
            post_sent_needed.append(utt_post[utt_id])
            seen_utt[utt_id.item()] = virtual_pointer
            virtual_pointer += 1
        virtual_address.append(seen_utt[utt_id.item()])
    return torch.cat(prev_sent_needed), torch.cat(post_sent_needed), torch.LongTensor(virtual_address)

def fill_uttemb_batch(utt_embeddings, embind, bsz, bptt):
    '''Fill current batch with corresponding utterances'''
    batched_utt_embeddings = torch.index_select(utt_embeddings, 0, embind)
    return batched_utt_embeddings.view(bptt, bsz, -1)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, ind, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    embind = ind[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, embind, target, seq_len

def get_batch_ngram(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    target = source[i+1:i+1+seq_len].view(-1)
    return target

def load_utt_embeddings(setname):
    return (torch.load(args.saveprefix+setname+'_utt_embed.pt'),
            torch.load(args.saveprefix+setname+'_fullind.pt'),
	    torch.load(args.saveprefix+setname+'_embind.pt'))

def batchify(data, embind, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    embind = embind.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    embind = embind.view(bsz, -1).t().contiguous()
    return data.to(device), embind

def batchify_ngram(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def fill_uttemb_batch(utt_embeddings, embind, bsz, bptt):
    '''Fill current batch with corresponding utterances'''
    batched_utt_embeddings = torch.index_select(utt_embeddings, 0, embind)
    return batched_utt_embeddings.view(bptt, bsz, -1)

def get_batch_emb(embeddings, data):
    batched_word_embeddings = torch.index_select(embeddings, 0, data.view(-1))
    return batched_word_embeddings.view(data.size(0), data.size(1), -1)

def debug_print_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)

def evaluate(evaldata, sent_ind_batched, utt_dict_prev, utt_dict_post, model,
             FLvmodel, ids_dict):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    model.set_mode('eval')
    FLvmodel.eval()
    FLvmodel.set_mode('eval')
    # Reset losses and counts
    total_loss = 0.
    total_words = 0.
    hidden = model.init_hidden(eval_batch_size)
    # Embedding tensor
    emb_size = FLvmodel.nhid
    prev_batched_embeddings = None
    post_batched_embeddings = None
    with torch.no_grad():
        for batch, i in enumerate(range(0, evaldata.size(0) - 1, args.bptt)):
            data, ind, targets, seq_len = get_batch(evaldata, sent_ind_batched, i)
            # check if the batch context idices are already filled
            if batch not in ids_dict:
                prev_utts, post_utts, ind_lookup = get_needed_utterance(
		    ind.view(-1), utt_dict_prev, utt_dict_post) 
                ids_dict[batch] = (prev_utts, post_utts, ind_lookup)
            else:
                prev_utts, post_utts, ind_lookup = ids_dict[batch]
            prev_utts_tensor = prev_utts.view(-1, max(1, args.maxlen_prev))
            post_utts_tensor = post_utts.view(-1, max(1, args.maxlen_post))
            FLvbatchsize = prev_utts_tensor.size(0)
            ind_lookup = ind_lookup.to(device)
            # Forward previous context information
            batched_embeddings = None
            if args.useatten:
                FLvhidden = FLvmodel.init_hidden(FLvbatchsize)
                if args.maxlen_prev != 0:
                    prev_embeddings = model.get_word_emb(prev_utts_tensor.t().contiguous().to(device))
                    prev_extracted, prevpenalty = FLvmodel(
		        prev_embeddings, FLvhidden, device, eosidx)
                else:
                    prev_extracted, prevpenalty = (
		        torch.zeros(FLvbatchsize, emb_size*args.nhead).to(device), 0)
                auxinput_prev = fill_uttemb_batch(prev_extracted, ind_lookup, eval_batch_size, seq_len)
                FLvhidden = FLvmodel.init_hidden(FLvbatchsize)
                if args.maxlen_post != 0:
                    post_embeddings = model.get_word_emb(post_utts_tensor.t().contiguous().to(device))
                    post_extracted, postpenalty = FLvmodel(
		        post_embeddings, FLvhidden, device, eosidx)
                else:
                    post_extracted, postpenalty = (
		        torch.zeros(FLvbatchsize, emb_size*args.nhead).to(device), 0)
                auxinput_post = fill_uttemb_batch(post_extracted, ind_lookup, eval_batch_size, seq_len)
                auxinput = torch.cat([auxinput_prev, auxinput_post], 2)

            # Here begins the forward path for second level LM
            output, hidden, penalty = model(
	        data, auxinput, hidden, eosidx=eosidx, device=device)
            output_flat = output.view(-1, ntokens)
            total_loss += criterion(output_flat, targets).data * len(data)
            total_words += len(data)
            hidden = repackage_hidden(hidden)
            
    return total_loss, total_words, ids_dict

def train(traindata, sent_ind_batched, utt_dict_prev, utt_dict_post, model,
          FLvmodel, ids_dict, epoch):
    """traindata: input data
       sent_ind_batched: sentence indices associated with the input data
       utt_dict_prev: previous utterance list
       utt_dict_post: future utterance list
       model: second level LM
       FLvmodel: first level LM
       ids_dict: processed batch cached
       epoch: current epoch number
    """
    total_loss = 0.
    total_penalty = 0.
    model.train()
    model.set_mode('train')
    FLvmodel.train()
    model.zero_grad()
    if epoch < 1 and not args.scratch: 
        logging('Not updating first level LM for this epoch!')
        FLvmodel.set_mode('eval')
    else:
        FLvmodel.set_mode('train')
        FLvmodel.zero_grad()
        FLvoptimizer = torch.optim.SGD(FLvmodel.parameters(), lr=FLlr, weight_decay=args.wdecay)
    hidden = model.init_hidden(args.batchsize)
    # Sentence embedding size
    emb_size = FLvmodel.nhid
    # Use SGD to optimize both LMs, can have different lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.wdecay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    prev_batched_embeddings = None
    post_batched_embeddings = None
    for batch, i in enumerate(range(0, traindata.size(0) - 1, args.bptt)):
        data, ind, targets, seq_len = get_batch(traindata, sent_ind_batched, i)
        # check if the batch context idices are already filled
        if batch not in ids_dict:
            prev_utts, post_utts, ind_lookup = get_needed_utterance(
                ind.view(-1), utt_dict_prev, utt_dict_post)
            ids_dict[batch] = (prev_utts, post_utts, ind_lookup)
        else:
            prev_utts, post_utts, ind_lookup = ids_dict[batch]
        prev_utts_tensor = prev_utts.view(-1, max(1, args.maxlen_prev))
        post_utts_tensor = post_utts.view(-1, max(1, args.maxlen_post))
        FLvbatchsize = prev_utts_tensor.size(0)
        ind_lookup = ind_lookup.to(device)
        # Forward previous context information

        batched_embeddings = None
        if args.useatten:
            FLvhidden = FLvmodel.init_hidden(FLvbatchsize)
            if args.maxlen_prev != 0:
                prev_embeddings = model.get_word_emb(prev_utts_tensor.t().contiguous().to(device))
                prev_extracted, prevpenalty = FLvmodel(prev_embeddings,
		                                       FLvhidden,
						       device=device,
						       eosidx=eosidx)
            else:
                prev_extracted, prevpenalty = (torch.zeros(FLvbatchsize, emb_size*args.nhead).to(device), 0)
            auxinput_prev = fill_uttemb_batch(prev_extracted, ind_lookup, args.batchsize, seq_len)
            FLvhidden = FLvmodel.init_hidden(FLvbatchsize)
            if args.maxlen_post != 0:
                post_embeddings = model.get_word_emb(post_utts_tensor.t().contiguous().to(device))
                post_extracted, postpenalty = FLvmodel(post_embeddings,
		                                       FLvhidden,
						       device=device,
						       eosidx=eosidx)
            else:
                post_extracted, postpenalty = (torch.zeros(FLvbatchsize, emb_size*args.nhead).to(device), 0)
            auxinput_post = fill_uttemb_batch(post_extracted, ind_lookup, args.batchsize, seq_len)
            auxinput = torch.cat([auxinput_prev, auxinput_post], 2)
            FLvpenalty = prevpenalty + postpenalty

        hidden = repackage_hidden(hidden)
        # Forward for the second level LM
        output, hidden, penalty = model(data, auxinput, hidden, eosidx=eosidx, device=device)

        loss = criterion(output.view(-1, ntokens), targets)

        if not args.useatten: 
            loss.backward()
        else:
            ploss = loss + args.alpha * FLvpenalty
            # import pdb; pdb.set_trace()
            ploss.backward()

        if FLvmodel.mode == 'train' and batch % args.updatedelay == 0:
            # Clip gradients for first level LM
            torch.nn.utils.clip_grad_value_(FLvmodel.parameters(), args.FLvclip)
            # Optimise only the first level LM
            FLvoptimizer.step()
            FLvmodel.zero_grad()
        elif FLvmodel.mode == 'eval':
            FLvmodel.zero_grad()
        if batch % args.updatedelay == 0:
            # Clip gradients for second level LM
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            # Optimise only the second level LM
            optimizer.step()
            model.zero_grad()

        total_loss += loss.item()
        total_penalty += args.alpha * FLvpenalty.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            cur_penalty = total_penalty / args.log_interval
            elapsed = time.time() - start_time
            logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | FLlr {:02.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | penalty {:2.2f}'.format(
                epoch, batch, traindata.size(0) // args.bptt, lr, FLlr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), float(cur_penalty)))
            total_loss = 0.
            total_penalty = 0.
            start_time = time.time()
    return model, FLvmodel, ids_dict

def loadNgram(path):
    probs = []
    with open(path) as fin:
        for line in fin:
            probs.append(float(line.strip()))
    return torch.Tensor(probs)

def display_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)

# ---------------------
# Main code starts here
# ---------------------
# Data loading
dictfile = os.path.join(args.data, 'dictionary.txt')

if args.use_sampling:
    train_loader, val_loader, test_loader, dictionary = L2joint_dataloader_atten.create(
        args.data, dictfile, batchSize=1, workers=0, maxlen_prev=args.maxlen_prev,
	maxlen_post=args.maxlen_post, use_sampling=True, errorfile=args.errorfile,
	reference=args.reference, ratio=args.ratio, random=args.randsample)
    train_loader.dataset.dictionary.use_sampling = False
    val_loader.dataset.dictionary.use_sampling = False
    test_loader.dataset.dictionary.use_sampling = False
else:
    train_loader, val_loader, test_loader, dictionary = L2joint_dataloader_atten.create(
        args.data, dictfile, batchSize=1, workers=0, maxlen_prev=args.maxlen_prev,
	maxlen_post=args.maxlen_post)
ntokens = len(dictionary.idx2word)
eosidx = dictionary.word2idx['<eos>']

# Model and optimizer instantiation
logging('Instantiating models and criteria')
FLvpretrained = torch.load(args.FLvmodel)
if not args.evalmode:
    if args.useatten:
        FLvmodel = AttenFlvModel(args.emsize, FLvpretrained.nhid, 1,
	                         args.nhid, args.dropout, nhead=args.nhead).to(device)
        FLvmodel.rnn.load_state_dict(FLvpretrained.rnn.state_dict())
        FLvmodel.rnn.flatten_parameters()
    elif args.scratch:
        FLvmodel = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers,
	                    args.dropout, args.dropout, reset=args.reset).to(device)
    else:
        FLvmodel = torch.load(args.FLvmodel)
        FLvmodel.rnn.flatten_parameters()
    # Here set atten flag to be False for 2nd level LM
    model = L2RNNModel(args.model, ntokens, args.emsize, FLvmodel.nhid, args.nhead*2,
                       args.naux, args.nhid, args.nlayers, False, args.dropout, reset=args.reset,
		       nhead=args.nhead).to(device)
criterion = nn.CrossEntropyLoss()
interpCrit = nn.CrossEntropyLoss(reduction='none')

# Start training
logging('Training Start!')
for pairs in arglist:
    logging(pairs[0] + ':  ' + str(pairs[1]))
# Loop over epochs.
best_val_loss = None
# tmp storage of utt indices for each training scp
train_ids_dict_list = {}
valid_ids_dict_list = {}
if not args.evalmode:
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            # iterate through scp minibatches
            if not args.use_sampling or epoch % args.sample_freq != 0:
                for i, train_batched in enumerate(train_loader):
                    # Check if the context for this batch is filled
                    if i not in train_ids_dict_list:
                        train_ids_dict_list[i] = {}
                    # iterate through scps in each minibatch, default is 1
                    for j, segment in enumerate(train_batched):
                        input_seg_file, sent_ind, sent_dict_prev, sent_dict_post = segment
                        data, sent_ind_batched = batchify(input_seg_file, sent_ind, args.batchsize)
                        # check for this particular scp whether context is filled
                        if j not in train_ids_dict_list[i]:
                            train_ids_dict_list[i][j] = {}
                        model, FLvmodel, train_ids_dict_list[i][j] = train(data,
			                                                   sent_ind_batched,
									   sent_dict_prev,
									   sent_dict_post,
									   model,
									   FLvmodel,
									   train_ids_dict_list[i][j],
									   epoch)
                logging('time elapsed is {:5.2f}s'.format((time.time() - epoch_start_time)))

            # Process an additional epoch for error sampling every epoch
            elif args.use_sampling:
                # Turn on error sampling
                train_loader.dataset.dictionary.use_sampling = True
                logging('Use error sampling, the sampled epoch starts here!')
                additional_epoch_start_time = time.time()
                for i, train_batched in enumerate(train_loader):
                    for j, segment in enumerate(train_batched):
                        input_seg_file, sent_ind, sent_dict_prev, sent_dict_post = segment
                        data, sent_ind_batched = batchify(input_seg_file, sent_ind, args.batchsize)
                        model, FLvmodel, _ = train(data,
			                           sent_ind_batched,
						   sent_dict_prev,
						   sent_dict_post,
						   model,
						   FLvmodel,
						   {},
						   epoch)
                logging('time elapsed is {:5.2f}s'.format((time.time() - additional_epoch_start_time)))
                # Turn off error sampling
                train_loader.dataset.dictionary.use_sampling = False

            # Process validation set
            aggregate_valloss = 0.
            total_valset = 0
            epoch_start_time = time.time()
            for i, val_batched in enumerate(val_loader):
                # Check if the context for this batch is filled
                if i not in valid_ids_dict_list:
                    valid_ids_dict_list[i] = {}
                for j, segment in enumerate(val_batched):
                    input_seg_file, sent_ind, sent_dict_prev, sent_dict_post = segment
                    data, sent_ind_batched = batchify(input_seg_file, sent_ind, eval_batch_size)
                    # check for this particular scp whether context is filled
                    if j not in valid_ids_dict_list[i]:
                        valid_ids_dict_list[i][j] = {}
                    val_loss, num_of_words, valid_ids_dict_list[i][j] = evaluate(data,
		                                                                 sent_ind_batched,
										 sent_dict_prev,
										 sent_dict_post,
										 model,
										 FLvmodel,
										 valid_ids_dict_list[i][j])
                    aggregate_valloss = aggregate_valloss + val_loss
                    total_valset += num_of_words
            val_loss = aggregate_valloss / total_valset
            logging('-' * 89)
            logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            logging('-' * 89)

            # Do the sampled validation
            if args.use_sampling:
                sampled_aggre_valloss = 0
                sampled_total = 0
                val_loader.dataset.dictionary.use_sampling = True
                for i, val_batched in enumerate(val_loader):
                    for j, segment in enumerate(val_batched):
                        input_seg_file, sent_ind, sent_dict_prev, sent_dict_post = segment
                        data, sent_ind_batched = batchify(input_seg_file, sent_ind, eval_batch_size)
                        sampled_val_loss, num_of_words, _ = evaluate(data,
			                                             sent_ind_batched,
								     sent_dict_prev,
								     sent_dict_post,
								     model,
								     FLvmodel,
								     {})
                        sampled_aggre_valloss = sampled_aggre_valloss + sampled_val_loss
                        sampled_total += num_of_words
                sampled_val_loss = sampled_aggre_valloss / sampled_total
                logging('-' * 41 + 'Sampled' + '-' * 41)
                logging('| end of epoch {:3d} | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, sampled_val_loss, math.exp(sampled_val_loss)))
                logging('-' * 89)
                val_loader.dataset.dictionary.use_sampling = False

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                with open(args.FLvsave, 'wb') as f:
                    torch.save(FLvmodel, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 2.0
                FLlr /= 2.0
    except KeyboardInterrupt:
        logging('-' * 89)
        logging('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    FLvmodel = torch.load(args.FLvsave)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()
    FLvmodel.rnn.flatten_parameters()

# Run on dev set again if in eval mode
if args.evalmode:
    aggregate_valloss = 0.
    total_valset = 0
    valid_ids_dict_list = {}
    for i, val_batched in enumerate(val_loader):
        # Check if the context for this batch is filled
        if i not in valid_ids_dict_list:
            valid_ids_dict_list[i] = {}
        for j, segment in enumerate(val_batched):
            input_seg_file, sent_ind, sent_dict_prev, sent_dict_post = segment
            data, sent_ind_batched = batchify(input_seg_file, sent_ind, eval_batch_size)
            # check for this particular scp whether context is filled
            if j not in valid_ids_dict_list[i]:
                valid_ids_dict_list[i][j] = {}
            val_loss, num_of_words, valid_ids_dict_list[i][j] = evaluate(data,
	                                                                 sent_ind_batched,
									 sent_dict_prev,
									 sent_dict_post,
									 model,
									 FLvmodel,
									 valid_ids_dict_list[i][j])
            aggregate_valloss = aggregate_valloss + val_loss
            total_valset += num_of_words
    val_loss = aggregate_valloss / total_valset
    logging('=' * 89)
    logging('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        val_loss, math.exp(val_loss)))
    logging('=' * 89)

# Run on test data.
aggregate_testloss = 0.
total_testset = 0
test_ids_dict_list = {} 
for i, test_batched in enumerate(test_loader):
    if i not in test_ids_dict_list:
        test_ids_dict_list[i] = {}
    for j, segment in enumerate(test_batched):
        input_seg_file, sent_ind, sent_dict_prev, sent_dict_post = segment
        data, sent_ind_batched = batchify(input_seg_file, sent_ind, eval_batch_size)
        if j not in test_ids_dict_list[i]:
            test_ids_dict_list[i][j] = {}
        test_loss, num_of_words, test_ids_dict_list[i][j] = evaluate(data,
	                                                             sent_ind_batched,
								     sent_dict_prev,
								     sent_dict_post,
								     model,
								     FLvmodel,
								     test_ids_dict_list[i][j])
        aggregate_testloss = aggregate_testloss + test_loss
        total_testset += num_of_words
test_loss = aggregate_testloss / total_testset
logging('=' * 89)
logging('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logging('=' * 89)
