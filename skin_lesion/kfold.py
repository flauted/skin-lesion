from random import shuffle as shuffle_
import itertools

import numpy as np

from torch.utils.data.sampler import Sampler


class KfoldSampler(Sampler):
    def __init__(self, n, k, shuffle=True):
        self.n = n
        self.k = k
        samples = list(range(self.n))
        if shuffle:
            shuffle_(samples)
        folds = np.array_split(samples, k)
        self.folds = folds
        self.i = 0
        self.train = True

    def next_fold(self):
        self.i += 1
        self.i %= self.k

    def val_fold(self):
        return self.folds[self.i]

    def train_folds(self):
        return itertools.chain.from_iterable(
            (self.folds[j] for j in range(self.k) if j != self.i))

    def __iter__(self):
        if self.train:
            return iter(self.train_folds())
        else:
            return iter(self.val_fold())

    def __len__(self):
        return self.n
