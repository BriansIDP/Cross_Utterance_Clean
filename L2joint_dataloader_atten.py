import sys, os
from torch.utils.data import Dataset, DataLoader
import torch

from ErrorSampling import ErrorSampling

class Dictionary(object):
    def __init__(self, dictfile, use_sampling=False, errorfile='', reference='', ratio=1, random=False):
        self.word2idx = {}
        self.idx2word = []
        self.build_dict(dictfile)
        self.use_sampling = use_sampling
        if use_sampling:
            self.sampler = ErrorSampling(dictfile, errorfile, reference, ratio, random)

    def build_dict(self, dictfile):
        with open(dictfile, 'r', encoding="utf8") as f:
            for line in f:
                index, word = line.strip().split(' ')
                self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def get_eos(self):
        return self.word2idx['<eos>']

    def get_sos(self):
        return self.word2idx['<sos>']

    def sent_to_idx(self, sentence):
        sent = []
        sampled_sent = []
        for word in sentence:
            to_append = self.word2idx[word] if word in self.word2idx else self.word2idx['OOV']
            sent.append(to_append)
	    # Randomly introduce acoustic errors
            if self.use_sampling:
                substitute = self.sampler.sample(word)
                substitutes = substitute.split()
                if len(substitutes) == 1:
                    sampled_sent.append(
		        self.word2idx[substitutes[0]] if substitutes[0] in self.word2idx else to_append)
                elif len(substitutes) == 2:
                    sampled_sent.append(
                        self.word2idx[substitutes[0]] if substitutes[0] in self.word2idx else to_append)
                    sampled_sent.append(
                        self.word2idx[substitutes[1]] if substitutes[1] in self.word2idx else 'OOV')
        if self.use_sampling:
            return sent, sampled_sent
        return sent, sent

    def __len__(self):
        return len(self.idx2word)


class LMdata(Dataset):
    def __init__(self, data_file, dictionary, maxlen_prev, maxlen_post):
        '''Load data_file'''
        self.data_file = data_file
        self.datascp = []
        with open(self.data_file, 'r') as f:
            for line in f:
                self.datascp.append(line.strip())
        self.dictionary = dictionary
        self.maxlen_prev = maxlen_prev
        self.maxlen_post = maxlen_post

    def __len__(self):
        return len(self.datascp)

    def __getitem__(self, idx):
        with open(self.datascp[idx], 'r') as fin:
            input_seg_file = []
            sent_ind = []
            sent_tank_prev = []
            sent_tank_post = []
            postlen = 0
            prevlen = 0
            sent_tank_mid = 0
            sent_dict_prev = []
            sent_dict_post = []
            sent_list = []
            eosidx = self.dictionary.word2idx['<eos>']
            # First run to read in sentences
            for i, line in enumerate(fin):
                idx_line, sampled_sent = self.dictionary.sent_to_idx(['<eos>']+line.split())
                input_seg_file += idx_line
                sent_ind += [i for j in range(len(idx_line))]
                sent_list.append(sampled_sent[1:])
            # Second run to get context
            for i, sent in enumerate(sent_list):
                sent_cursor = i - 1
                sent_tank_prev = []
                sent_tank_post = []
                while len(sent_tank_prev) <= self.maxlen_prev and sent_cursor >= 0:
                    sent_tank_prev = sent_list[sent_cursor] + sent_tank_prev
                    sent_cursor -= 1
                if len(sent_tank_prev) < self.maxlen_prev:
                    sent_tank_prev = [eosidx] * (self.maxlen_prev - len(sent_tank_prev)) + sent_tank_prev
                elif self.maxlen_prev == 0:
                    sent_tank_prev = [eosidx]
                else:
                    sent_tank_prev = sent_tank_prev[-self.maxlen_prev:]
                sent_cursor = i + 1
                while len(sent_tank_post) <= self.maxlen_post and sent_cursor < len(sent_list):
                    sent_tank_post += sent_list[sent_cursor]
                    sent_cursor += 1
                if len(sent_tank_post) < self.maxlen_post:
                    sent_tank_post = [eosidx] * (self.maxlen_post - len(sent_tank_post)) + sent_tank_post
                elif self.maxlen_post == 0:
                    sent_tank_post = [eosidx]
                else:
                    sent_tank_post = sent_tank_post[:self.maxlen_post]
                sent_dict_prev.append(torch.LongTensor(sent_tank_prev).view(1,-1))
                sent_dict_post.append(torch.LongTensor(sent_tank_post).view(1,-1))
            sent_dict_prev = torch.cat(sent_dict_prev)
            sent_dict_post = torch.cat(sent_dict_post)
        return (torch.LongTensor(input_seg_file), torch.LongTensor(sent_ind), sent_dict_prev, sent_dict_post)

def collate_fn(batch):
    return [f for f in batch]

def create(datapath, dictfile, batchSize=1,
           shuffle=False, workers=0, maxlen_prev=30,
	   maxlen_post=30, use_sampling=False, errorfile='', reference='',
           ratio=1, random=False):
    loaders = []
    dictionary = Dictionary(dictfile, use_sampling, errorfile, reference, ratio, random)
    for split in ['train', 'valid', 'test']:
        data_file = os.path.join(datapath, '%s.scp' %split)
        dataset = LMdata(data_file, dictionary, maxlen_prev, maxlen_post)
        loaders.append(DataLoader(dataset=dataset, batch_size=batchSize,
                                  shuffle=shuffle, collate_fn=collate_fn,
                                  num_workers=workers))
    return loaders[0], loaders[1], loaders[2], dictionary

if __name__ == "__main__":
    datapath = sys.argv[1]
    dictfile = sys.argv[2]
    errorfile = sys.argv[3]
    reference = sys.argv[4]
    traindata, valdata, testdata, dictionary = create(datapath, dictfile, batchSize=1, workers=0, use_sampling=True, errorfile=errorfile, reference=reference)
    for i_batch, sample_batched in enumerate(traindata):
        print(i_batch, len(sample_batched))
