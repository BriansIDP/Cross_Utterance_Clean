import sys, os
from torch.utils.data import Dataset, DataLoader
import torch

class Dictionary(object):
    def __init__(self, dictfile):
        self.word2idx = {}
        self.idx2word = []
        self.unigram = []
        self.build_dict(dictfile)

    def build_dict(self, dictfile):
        with open(dictfile, 'r', encoding="utf8") as f:
            for line in f:
                index, word = line.strip().split(' ')
                self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.unigram.append(0)
        self.unigram[self.word2idx[word]] += 1
        return self.word2idx[word]

    def get_eos(self):
        return self.word2idx['<eos>']

    def normalize_counts(self):
        self.unigram /= np.sum(self.unigram)
        self.unigram = self.unigram.tolist()

    def __len__(self):
        return len(self.idx2word)

class LMdata(Dataset):
    def __init__(self, filelist, dictionary):
        '''Load data_file'''
        self.files = []
        with open(filelist, 'r') as f:
            for line in f:
                self.files.append(line.strip())
        self.dictionary = dictionary

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        word_ind = []
        with open(self.files[idx], 'r') as fin:
            for line in fin:
                words = line.split() + ['<eos>']
                for word in words:
                    if word in self.dictionary.word2idx:
                        word_ind.append(self.dictionary.word2idx[word])
                    else:
                        word_ind.append(self.dictionary.word2idx['OOV'])
        return word_ind

def collate_fn(batch):
    return torch.LongTensor(batch).view(-1)

def create(datapath, batchSize=1, shuffle=False, workers=0):
    loaders = []
    dictfile = os.path.join(datapath, 'dictionary.txt')
    dictionary = Dictionary(dictfile)
    for split in ['train', 'valid', 'test']:
        data_file = os.path.join(datapath, '%s.scp' %split)
        dataset = LMdata(data_file, dictionary)
        loaders.append(DataLoader(dataset=dataset, batch_size=batchSize,
                                  shuffle=shuffle, collate_fn=collate_fn,
                                  num_workers=workers))
    return loaders[0], loaders[1], loaders[2]

if __name__ == "__main__":
    datapath = sys.argv[1]
    traindata, valdata, testdata = create(datapath, batchSize=1, workers=0)
    for i_batch, sample_batched in enumerate(traindata):
        print(i_batch, sample_batched.size())
