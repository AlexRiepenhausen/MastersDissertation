import torch
import numpy as np
from torch.utils.data import Dataset

class VectorDataset(Dataset):

    def __init__(self, file_paths, label_file, seq_dim, batch_size=1):
        self.file_paths = file_paths
        self.num_files  = len(file_paths)
        self.batch_size = batch_size
        self.labels     = self.initLabels(label_file)
        self.seq_dim    = seq_dim


    # save labels of every document so that they are readily accessible
    def initLabels(self, label_file):

        labels = list()
        file = open(label_file, 'r', encoding='utf8')
        for line in file:
            item = line.replace("\n","")
            labels.append(int(item))

        return np.asarray(labels)


    def __len__(self):
        return self.num_files/self.batch_size


    def __getitem__(self, idx):

        randindex = np.random.randint(low=0, high=self.num_files)
        file   = open(self.file_paths[randindex], 'r', encoding='utf8')

        vectors   = list()
        for i in range(0, self.seq_dim):
            arr = np.fromstring(file.readline(), dtype=float, sep=" ")
            vectors.append(arr)

        return torch.tensor([np.asarray(vectors)]).float(), torch.tensor([self.labels[randindex]]).long()

