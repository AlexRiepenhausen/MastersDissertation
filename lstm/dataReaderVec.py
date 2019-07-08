import torch
import random
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict

class VectorDataset(Dataset):

    def __init__(self, file_paths, labels, seq_dim, batch_size=1):
        self.file_paths = file_paths
        self.num_files  = len(file_paths)
        self.batch_size = batch_size
        self.labels     = np.asarray(labels)
        self.seq_dim    = seq_dim

        # print label distribution
        self.labelHistogram()


    def __len__(self):
        return self.num_files/self.batch_size


    def labelHistogram(self):
        lbl_hist = defaultdict(int)
        for label in self.labels:
            lbl_hist[label] += 1

        lbl_hist = OrderedDict(sorted(lbl_hist.items()))
        print("| ---- Label Frequency ----  |")
        print("|                            |")
        for item in lbl_hist:
                print("| Label: {:2d}, Frequency: {:4d} |".format(item, lbl_hist[item]))
        print("|                            |")


    def __getitem__(self, idx):

        findex = random.randint(-1, self.num_files-1)
        file   = open(self.file_paths[findex], 'r', encoding='utf8')

        vectors  = list()
        line = file.readline()

        while line:
            arr = np.fromstring(line, dtype=float, sep=" ")
            vectors.append(arr)
            line = file.readline()

        return torch.tensor([np.asarray(vectors)]).float(), torch.tensor([self.labels[findex]]).long()

