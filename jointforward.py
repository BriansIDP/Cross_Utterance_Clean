# coding: utf-8
import argparse
import sys, os
import torch
import math
from operator import itemgetter

import data

parser = argparse.ArgumentParser(description='PyTorch Level-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/AMI',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='model.pt',
                    help='location of the 2nd level model')
parser.add_argument('--FLvmodel', type=str, default='FLvmodel.pt',
                    help='location of the 1st level model')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--reset', action='store_true',
                    help='reset at sentence boundaries')
parser.add_argument('--memorycell', action='store_true',
                    help='Use memory cell as input, otherwise use output cell')
parser.add_argument('--rnnscale', type=float, default=6,
                    help='how much importance to attach to rnn score')
parser.add_argument('--lm', type=str, default='original',
                    help='Specify which language model to be used: rnn, ngram or original')
parser.add_argument('--nbest', type=str, default='dev.nbest.info.txt',
                    help='Specify which nbest file to be used')
parser.add_argument('--ngram', type=str, default='dev_ngram.st',
                    help='Specify which ngram stream file to be used')
parser.add_argument('--saveemb', action='store_true',
                    help='save utterance embeddings')
parser.add_argument('--context', type=str, default='0',
                    help='Specify which utterance embeddings to be used')
parser.add_argument('--interp', action='store_true',
                    help='Linear interpolation of LMs')
parser.add_argument('--useatten', action='store_true',
                    help='Use self-attention based LM')
parser.add_argument('--factor', type=float, default=0.8,
                    help='ngram interpolation weight factor')
parser.add_argument('--gscale', type=float, default=12.0,
                    help='ngram grammar scaling factor')
parser.add_argument('--logfile', type=str, default='resnet_nbest/log.txt',
                    help='Forward log file')
parser.add_argument('--outputcell', type=int, default=1,
                    help='which hidden state to be used')
parser.add_argument('--arrange', type=str, default='sentence',
                    help='Arrangements: sentence, segments or attention')
parser.add_argument('--maxlen', type=int, default=36,
                    help='No. of words to look at')
parser.add_argument('--seglen', type=int, default=20,
                    help='No. of words to look at')
parser.add_argument('--overlap', type=int, default=0,
                    help='No. of word overlap between 2 segments')
parser.add_argument('--sepchunk', action='store_true',
                    help='Use separate RNNs for segments')
args = parser.parse_args()

def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(args.logfile, 'a+') as f_log:
            f_log.write(s + '\n')

# Read in dictionary
logging("Reading dictionary...")
dictionary = {}
with open(os.path.join(args.data, 'dictionary.txt')) as vocabin:
    lines = vocabin.readlines()
    for line in lines:
        ind, word = line.strip().split(' ')
        if word not in dictionary:
            dictionary[word] = ind
        else:
            logging("Error! Repeated words in the dictionary!")

ntokens = len(dictionary)
eosidx = int(dictionary['<eos>'])
context_shift = [int(i) for i in args.context.strip().split()]
device = torch.device("cuda" if args.cuda else "cpu")

def readin_model():
    # Read in trained 1st level model
    logging("Reading model...")
    with open(args.model, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()
        if args.cuda:
            model.cuda()
    return model

def readin_FLvmodel():
    logging("Reading FLvmodel...")
    with open(args.FLvmodel, 'rb') as f:
        FLvmodel = torch.load(f)
        FLvmodel.rnn.flatten_parameters()
        if args.cuda:
            FLvmodel.cuda()
    return FLvmodel

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_chunks(utts):
    batchsize = utts.size(0)
    expanded_utts = []
    cursor = 0
    while cursor+args.seglen <= args.maxlen:
        expanded_utts.append(utts[cursor:cursor+args.seglen].view(1,-1))
        cursor = cursor + args.seglen - args.overlap
    return torch.cat(expanded_utts).to(device)

def get_nseg():
    nsegments = 0
    segcursor = 0
    while segcursor + args.seglen <= args.maxlen:
        segcursor = segcursor + args.seglen - args.overlap
        nsegments += 1
    return nsegments

def FLvForwarding(infile, FLvmodel):
    '''Forward first level LM to get sentence embeddings'''
    logging('Start forwarding the first level LM')
    sentdict = {}
    count = 0
    with open(infile) as fin:
        for i, line in enumerate(fin):
            currentline = []
            linevec = line.strip().split()
            for j, word in enumerate(linevec):
                if word in dictionary:
                    currentline.append(int(dictionary[word]))
                else:
                    currentline.append(int(dictionary['OOV']))
            input = torch.LongTensor(currentline).to(device).view(1, -1).t()
            hidden = FLvmodel.init_hidden(1)
            output, hidden = FLvmodel(input, hidden, outputflag=1)
            half = len(currentline) // 2
            if args.outputcell == 0:
                sentdict[i] = hidden[1].view(-1)
            elif args.outputcell == 1:
                sentdict[i] = output[-1].view(-1)
            elif args.outputcell == 2:
                sentdict[i] = torch.cat([output[half].view(-1), output[-1].view(-1)])
            count += 1
            if count % 1000 == 0:
                logging('First level model forward finished {:5d}'.format(count))
    return sentdict, count

def FLvFixedForwarding(infile, FLvmodel):
    '''Forward first level LM to get segment level embeddings'''
    logging('Start forwarding the first level LM')
    sent_list = []
    sentdict = {}
    with open(infile) as fin:
        for i, line in enumerate(fin):
            currentline = []
            linevec = line.strip().split()
            for j, word in enumerate(linevec[:-1]):
                if word in dictionary:
                    currentline.append(int(dictionary[word]))
                else:
                    currentline.append(int(dictionary['OOV']))
            sent_list.append(currentline)
        # Second run to get context
        for i, sent in enumerate(sent_list):
            sent_cursor = i - 1
            sent_tank_prev = []
            sent_tank_post = []
            while len(sent_tank_prev) <= args.maxlen and sent_cursor >= 0:
                sent_tank_prev = sent_list[sent_cursor] + sent_tank_prev
                sent_cursor -= 1
            if len(sent_tank_prev) <= args.maxlen:
                sent_tank_prev = [eosidx] * (args.maxlen - len(sent_tank_prev)) + sent_tank_prev
            else:
                sent_tank_prev = sent_tank_prev[-args.maxlen:]
            sent_cursor = i + 1
            while len(sent_tank_post) <= args.maxlen and sent_cursor < len(sent_list):
                sent_tank_post += sent_list[sent_cursor]
                sent_cursor += 1
            if len(sent_tank_post) <= args.maxlen:
                sent_tank_post += [eosidx] * (args.maxlen - len(sent_tank_post))
            else:
                sent_tank_post = sent_tank_post[:args.maxlen]
            # Forward past segments
            input_prev = torch.LongTensor(sent_tank_prev).to(device).view(1, -1).t()
            hidden = FLvmodel.init_hidden(1)
            output_prev, hidden = FLvmodel(input_prev, hidden, outputflag=1)
            # Forward future segments
            input_post = torch.LongTensor(sent_tank_post).to(device).view(1, -1).t()
            hidden = FLvmodel.init_hidden(1)
            output_post, hidden = FLvmodel(input_post, hidden, outputflag=1)
            contexttensor = []
            tensorsize = int(args.maxlen / args.seglen)
            for j in range(tensorsize):
                contexttensor.append(output_prev[(j+1)*args.seglen-1].view(-1))
            for j in range(tensorsize):
                contexttensor.append(output_post[(j+1)*args.seglen-1].view(-1))
            sentdict[i] = torch.cat(contexttensor)
            if i % 1000 == 0:
                logging('first level completed: ' + str(i))
    return sentdict

def FLvSegOverlapForwarding(infile, FLvmodel):
    '''Forward first level LM to get segment level embeddings'''
    logging('Start forwarding the first level LM')
    sent_list = []
    sentdict = {}
    with open(infile) as fin:
        for i, line in enumerate(fin):
            currentline = []
            linevec = line.strip().split()
            for j, word in enumerate(linevec[1:-1]):
                if word in dictionary:
                    currentline.append(int(dictionary[word]))
                else:
                    currentline.append(int(dictionary['OOV']))
            sent_list.append(currentline)
        # Second run to get context
        for i, sent in enumerate(sent_list):
            sent_cursor = i - 1
            sent_tank_prev = []
            sent_tank_post = []
            while len(sent_tank_prev) <= args.maxlen and sent_cursor >= 0:
                sent_tank_prev = sent_list[sent_cursor] + sent_tank_prev
                sent_cursor -= 1
            if len(sent_tank_prev) <= args.maxlen:
                sent_tank_prev = [eosidx] * (args.maxlen - len(sent_tank_prev)) + sent_tank_prev
            else:
                sent_tank_prev = sent_tank_prev[-args.maxlen:]
            sent_cursor = i + 1
            while len(sent_tank_post) <= args.maxlen and sent_cursor < len(sent_list):
                sent_tank_post += sent_list[sent_cursor]
                sent_cursor += 1
            if len(sent_tank_post) <= args.maxlen:
                sent_tank_post += [eosidx] * (args.maxlen - len(sent_tank_post))
            else:
                sent_tank_post = sent_tank_post[:args.maxlen]
            # Arranging the overlaped segments into parallel streams
            prev_utts = get_chunks(torch.LongTensor(sent_tank_prev))
            post_utts = get_chunks(torch.LongTensor(sent_tank_post))
            # And then forward them
            FLvhidden = FLvmodel.init_hidden(prev_utts.size(0))
            FLvoutput, FLvhidden = FLvmodel(prev_utts.t(), FLvhidden, outputflag=1)
            utt_embeddings_prev = FLvoutput[-1].view(-1)
            FLvhidden = FLvmodel.init_hidden(post_utts.size(0))
            FLvoutput, FLvhidden = FLvmodel(post_utts.t(), FLvhidden, outputflag=1)
            utt_embeddings_post = FLvoutput[-1].view(-1)
            sentdict[i] = torch.cat([utt_embeddings_prev, utt_embeddings_post])
            if i % 1000 == 0:
                logging('first level completed: ' + str(i))
    return sentdict

def FLvAttenForwarding(infile, FLvmodel):
    '''Forward first level LM to get segment level embeddings'''
    logging('Start forwarding the first level LM')
    sent_list = []
    sentdict = {}
    with open(infile) as fin:
        for i, line in enumerate(fin):
            currentline = []
            linevec = line.strip().split()
            for j, word in enumerate(linevec[1:-1]):
                if word in dictionary:
                    currentline.append(int(dictionary[word]))
                else:
                    currentline.append(int(dictionary['OOV']))
            sent_list.append(currentline)
        # Second run to get context
        for i, sent in enumerate(sent_list):
            sent_cursor = i - 1
            sent_tank_prev = []
            sent_tank_post = []
            while len(sent_tank_prev) <= args.maxlen and sent_cursor >= 0:
                sent_tank_prev = sent_list[sent_cursor] + sent_tank_prev
                sent_cursor -= 1
            if len(sent_tank_prev) <= args.maxlen:
                sent_tank_prev = [eosidx] * (args.maxlen - len(sent_tank_prev)) + sent_tank_prev
            else:
                sent_tank_prev = sent_tank_prev[-args.maxlen:]
            sent_cursor = i + 1
            while len(sent_tank_post) <= args.maxlen and sent_cursor < len(sent_list):
                sent_tank_post += sent_list[sent_cursor]
                sent_cursor += 1
            if len(sent_tank_post) <= args.maxlen:
                sent_tank_post += [eosidx] * (args.maxlen - len(sent_tank_post))
            else:
                sent_tank_post = sent_tank_post[:args.maxlen]
            # Start forwarding
            input_prev = torch.LongTensor(sent_tank_prev).to(device).view(1, -1).t()
            FLvhidden = FLvmodel.init_hidden(1)
            prev_extracted, prevpenalty = FLvmodel(input_prev, FLvhidden, device=device)
            input_post = torch.LongTensor(sent_tank_post).to(device).view(1, -1).t()
            FLvhidden = FLvmodel.init_hidden(1)
            post_extracted, postpenalty = FLvmodel(input_post, FLvhidden, device=device)
            sentdict[i] = torch.cat([prev_extracted, post_extracted], 1)
            if i % 1000 == 0:
                logging('first level completed: ' + str(i))
    return sentdict

def SharedFLvAttenForwarding(infile, FLvmodel, model):
    '''Forward first level LM to get segment level embeddings'''
    logging('Start forwarding the first level LM')
    sent_list = []
    sentdict = {}
    with open(infile) as fin:
        for i, line in enumerate(fin):
            currentline = []
            linevec = line.strip().split()
            for j, word in enumerate(linevec[1:-1]):
                if word in dictionary:
                    currentline.append(int(dictionary[word]))
                else:
                    currentline.append(int(dictionary['OOV']))
            sent_list.append(currentline)
        # Second run to get context
        for i, sent in enumerate(sent_list):
            sent_cursor = i - 1
            sent_tank_prev = []
            sent_tank_post = []
            while len(sent_tank_prev) <= args.maxlen and sent_cursor >= 0:
                sent_tank_prev = sent_list[sent_cursor] + sent_tank_prev
                sent_cursor -= 1
            if len(sent_tank_prev) <= args.maxlen:
                sent_tank_prev = [eosidx] * (args.maxlen - len(sent_tank_prev)) + sent_tank_prev
            else:
                sent_tank_prev = sent_tank_prev[-args.maxlen:]
            sent_cursor = i + 1
            while len(sent_tank_post) <= args.maxlen and sent_cursor < len(sent_list):
                sent_tank_post += sent_list[sent_cursor]
                sent_cursor += 1
            if len(sent_tank_post) <= args.maxlen:
                sent_tank_post += [eosidx] * (args.maxlen - len(sent_tank_post))
            else:
                sent_tank_post = sent_tank_post[:args.maxlen]
            # Start forwarding
            input_prev = torch.LongTensor(sent_tank_prev).to(device).view(1, -1).t().contiguous()
            prev_emb = model.get_word_emb(input_prev)
            FLvhidden = FLvmodel.init_hidden(1)
            prev_extracted, prevpenalty = FLvmodel(prev_emb, FLvhidden, device=device)
            input_post = torch.LongTensor(sent_tank_post).to(device).view(1, -1).t().contiguous()
            post_emb = model.get_word_emb(input_post)
            FLvhidden = FLvmodel.init_hidden(1)
            post_extracted, postpenalty = FLvmodel(post_emb, FLvhidden, device=device)
            sentdict[i] = torch.cat([prev_extracted, post_extracted], 1)
            if i % 1000 == 0:
                logging('first level completed: ' + str(i))
    return sentdict

# Forward each sentence in the nbest list
def forward_each_utterance(model, line, forwardCrit, utt_idx, ngram_probs, aux_in, hidden):
    # hidden = model.init_hidden(1)
    linevec = line.strip().split()
    acoustic_score = float(linevec[0])
    lmscore = float(linevec[1])
    # Not sure if we need <eos> at the beginning of the sequence
    utterance = linevec[4:-1]
    currentline = []
    for i, word in enumerate(utterance):
        if word in dictionary:
            currentline.append(int(dictionary[word]))
        else:
            currentline.append(int(dictionary['OOV']))
    currentline = [eosidx] + currentline
    currenttarget = currentline[1:]
    currenttarget.append(eosidx)
    targets = torch.LongTensor(currenttarget).to(device)
    input = torch.LongTensor(currentline).to(device)
    input = input.view(1, -1).t()
    n = input.size(0)
    # Expand the auxiliary input feature
    aux_in = aux_in.repeat(n, 1).view(n, 1, -1)
    output, hidden, penalty = model(input, aux_in, hidden, eosidx=eosidx, device=device)
    logProb = forwardCrit(output.view(-1, ntokens), targets)
    if args.interp:
        ngram_probs = torch.tensor([float(prob)/args.gscale for prob in ngram_probs])
        log_prob_ngram = ngram_probs.to(device)
        rnnProbs = torch.exp(log_prob_ngram) * args.factor + torch.exp(-logProb) * (1 - args.factor)
        rnnscore = - float(torch.log(rnnProbs).sum())
    else:
        rnnscore = float(logProb * len(currentline))
    # Calculate total score
    total_score = - rnnscore * args.rnnscale + acoustic_score
    out = '\t'.join([str(utt_idx), str(acoustic_score), '{:5.2f}'.format(rnnscore), '{:5.2f}'.format(total_score), ' '.join(utterance)+' <eos>\n'])
    return out, total_score, utterance, hidden

def forward_nbest_utterance(model, FLvmodel, nbestfile):
    logging('Start calculating language model scores')
    model.eval()
    model.set_mode('eval')
    prev_hid = model.init_hidden(1)
    FLvmodel.eval()
    FLvmodel.set_mode('eval')
    if args.interp:
        forwardCrit = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        forwardCrit = torch.nn.CrossEntropyLoss()
    ngram_cursor = 0
    to_write = []
    best_utt_list = []
    emb_list = []
    utt_idx = 0
    ngram_probs = []
    # Ngram used for lattice rescoring
    ngram_listfile = open(args.ngram)
    # get context sentences
    with torch.no_grad():
        if args.arrange == 'sentence':
            sent_dict, totalutt = FLvForwarding(nbestfile+'.context', FLvmodel)
        elif args.arrange == 'segment':
            if args.overlap == 0:
                sent_dict = FLvFixedForwarding(nbestfile+'.context', FLvmodel)
            else:
                sent_dict = FLvSegOverlapForwarding(nbestfile+'.context', FLvmodel)
        elif args.arrange == 'attention':
            sent_dict = FLvAttenForwarding(nbestfile+'.context', FLvmodel)
        elif args.arrange == 'atten_shared':
            sent_dict = SharedFLvAttenForwarding(nbestfile+'.context', FLvmodel, model)
    with open(nbestfile) as filein:
        with torch.no_grad():
            for utterancefile in filein:
                labname = utterancefile.strip().split('/')[-1]
                labname = labname + '.rec'
                # Fill in contexts for utterance embeddings indexing
                if args.arrange == 'sentence':
                    current_context = []
                    for i in context_shift:
                        if i + utt_idx < 0:
                            current_context.append(sent_dict[0])
                        elif i + utt_idx >= totalutt-1:
                            current_context.append(sent_dict[totalutt-1])
                        else:
                            current_context.append(sent_dict[i+utt_idx])
                    current_aux_in = torch.cat(current_context)
                elif args.arrange in ['segment', 'attention', 'atten_shared']:
                    current_aux_in = sent_dict[utt_idx]
                # Load ngram probability file
                ngram_probfile_name = ngram_listfile.readline()
                ngram_probfile = open(ngram_probfile_name.strip())
                ngram_prob_lines = ngram_probfile.readlines()
                with open(utterancefile.strip()) as uttfile:
                    uttlines = uttfile.readlines()
                uttscore = []
                for i, line in enumerate(uttlines):
                    if args.interp:
                        ngram_elems = ngram_prob_lines[i].strip().split(' ')
                        sent_len = int(ngram_elems[0])
                        ngram_probs = ngram_elems[sent_len+2:]
                    outputline, score, utt, hid = forward_each_utterance(model, line, forwardCrit, utt_idx, ngram_probs, current_aux_in, prev_hid)
                    to_write.append(outputline)
                    uttscore.append((utt, score, hid))
                utt_idx += 1
                bestutt_group = max(uttscore, key=itemgetter(1))
                bestutt = bestutt_group[0]
                prev_hid = bestutt_group[2]
                best_utt_list.append((labname, bestutt))
                if utt_idx % 100 == 0:
                    logging(str(utt_idx))
                # print(utt_idx)
    with open(nbestfile+'.renew.'+args.lm, 'w') as fout:
        fout.writelines(to_write)
    with open(nbestfile + '.1best.'+args.lm, 'w') as fout:
        fout.write('#!MLF!#\n')
        for eachutt in best_utt_list:
            labname = eachutt[0]
            start = 100000
            end = 200000
            fout.write('\"'+labname+'\"\n')
            for eachword in eachutt[1]:
                if eachword[0] == '\'':
                    eachword = '\\' + eachword
                fout.write(str(start) + ' ' + str(end) + ' ' + eachword+'\n')
                start += 100000
                end += 100000
            fout.write('.\n')

# Main code begins
model = readin_model()
FLvmodel = readin_FLvmodel()
print('getting utterances')
forward_nbest_utterance(model, FLvmodel, args.nbest)
