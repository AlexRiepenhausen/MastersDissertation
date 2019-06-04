import torch
import numpy as np
from torch.utils.data import Dataset
from utilities import utilities

class VectorDataset(Dataset):

    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.num_files  = len(file_paths)

    def __len__(self):
        return 20

    def __getitem__(self, idx):

        findex = np.random.randint(low=0, high=self.num_files)
        file   = open(self.file_paths[findex], 'r', encoding='utf8')

        vectors   = list()
        for line in file:
            arr = np.fromstring(line.replace("['","").replace("']","").replace("\n",""), dtype=float, sep="', '")
            vectors.append(arr)

        return torch.tensor([np.asarray(vectors)]).double()

