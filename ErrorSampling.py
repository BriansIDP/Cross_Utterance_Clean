"""
This is the error sampling class whose input is the confusion file
Generates an sampler object which has pmf of acoustic errors for each word
Frequency-based unigram
"""
import sys, os
from collections import defaultdict

import numpy as np

class ErrorSampling():
    def __init__(self, dictionary, errorfile, hypothesis):
        self.errorfile = open(errorfile)
        self.dictionary = self.build_dict(dictionary)
        self.unigram = self.build_unigram(hypothesis)
        self.build_confusion()

    def build_dict(self, dictfile):
        dictionary = {}
        with open(dictfile) as fin:
            for line in fin:
                index, word = line.strip().split(' ')
                dictionary[word] = {'alternatives':[], 'probabilities':[]}
        return dictionary

    def build_unigram(self, hypothesis):
        wordcount = defaultdict(int)
        with open(hypothesis) as fin:
            for line in fin:
                for word in line.split():
                    if '\\' in word:
                        word = word.replace("\\", "")
                    if word in self.dictionary:
                        wordcount[word] += 1
        return wordcount

    def build_confusion(self):
        current = ''
        for line in self.errorfile:
            elems = line.split()
            if elems != [] and elems[0] == 'CONFUSION':
                current = 'CONFUSION'
            elif elems != [] and elems[0] == 'INSERTIONS':
                break
            if current != '' and elems != [] and ':' in elems[0]:
                key = elems[3].upper()
                value = elems[5].upper()
                if key in self.unigram:
                    if value in self.dictionary:
                        self.dictionary[key]['alternatives'].append(value)
                        self.dictionary[key]['probabilities'].append(float(elems[1]))
                    else:
                        if 'OOV' in self.dictionary[key]['alternatives']:
                            OOVind = self.dictionary[key]['alternatives'].index('OOV')
                            self.dictionary[key]['probabilities'][OOVind] += float(elems[1])
                        else:
                            self.dictionary[key]['alternatives'].append('OOV')
                            self.dictionary[key]['probabilities'].append(float(elems[1]))
        for key, value in self.dictionary.items():
            # Confusion pairs may contain more than what appeared in ref
            # TODO: need to check why this is the case, for now we bias this estimation
            self.dictionary[key]['alternatives'].append(key)
            self.dictionary[key]['probabilities'].append(self.unigram[key]/5 if self.unigram[key] != 0 else 1)
            total = sum(self.dictionary[key]['probabilities'])
            self.dictionary[key]['probabilities'] = [prob / total for prob in self.dictionary[key]['probabilities']]

    def sample(self, word):
        substitute = word
        if word in self.dictionary:
            distribution = self.dictionary[word]['probabilities']
            alternatives = self.dictionary[word]['alternatives']
            substitute = np.random.choice(alternatives, p=distribution)
        return substitute

if __name__ == "__main__":
    dictionary = sys.argv[1]
    errorfile = sys.argv[2]
    hypothesis = sys.argv[3]
    sampler = ErrorSampling(dictionary, errorfile, hypothesis)
    import pdb; pdb.set_trace()
