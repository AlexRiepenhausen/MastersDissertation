import torch
import numpy as np
import random
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict

class VectorDataset(Dataset):

    def __init__(self, file_paths, labels, loadertype="train", batch_size=1):
    
        self.loadertype = loadertype
    
        self.file_paths = file_paths
        self.num_files  = len(file_paths)
        self.batch_size = batch_size
        
        self.keywords   = self.getKeywords()
        self.labels     = self.getLabelsFromFiles(labels)    
        self.files      = self.readFiles()    
        
        self.index      = 0


    def __len__(self):
        return self.num_files/self.batch_size
        
  
    def getKeywords(self):
        path = './data/w2v/training/dictionary/house_flat.txt'
        keywords = dict()
        label = 0
        for line in open(path, encoding="utf8"):
            keywords[line.replace('\n','')] = label #0,1,2,3 ...
            label += 1
        return keywords
        
        
    def getLabelsFromFiles(self, files):
    
        labels = list() #list of lists
        for file in files:
            lines = list()
            for line in open(file, encoding="utf8"):
                lines.append(line.replace('\n',''))    
            labels.append(lines)
    
        return labels
        
        
    def readFiles(self):
    
        allfiles = list()
    
        for index in range(0,self.num_files):
        
            vectorfile = open(self.file_paths[index], 'r', encoding='utf8')
    
            vectors = list()
            line    = vectorfile.readline()
      
            while line:
                arr = np.fromstring(line, dtype=float, sep=" ")
                vectors.append(arr)
                line = vectorfile.readline()  
                
            allfiles.append(vectors) 
            
        return allfiles   
   
   
    def house_or_flat(self, label):
    
        if label != 'house': 
            if label != 'flat':
                return False
        
        if label != 'flat': 
            if label != 'house':
                return False
                
        return True
                                

    def __getitem__(self, idx):
    
        if self.loadertype == "train":
     
            index      = random.randint(0, self.num_files-1)      
            label_str  = self.labels[index]
            
            while not self.house_or_flat(label_str[0]):
                index      = random.randint(0, self.num_files-1)  
                label_str  = self.labels[index]
                 
            label      = self.keywords[label_str[0]]      
            vectorfile = self.files[index]
    
            return torch.tensor([np.asarray(vectorfile)]).float(), torch.tensor(label).long()
            
        if self.loadertype == "test":
        
            if self.index >= 100:
                self.index = 0    
        
            label_str  = self.labels[self.index]
            
            while not self.house_or_flat(label_str[0]):
                self.index += 1
                label_str  = self.labels[self.index]
                 
            label      = self.keywords[label_str[0]]      
            vectorfile = self.files[self.index]
            
            self.index += 1
    
            return torch.tensor([np.asarray(vectorfile)]).float(), torch.tensor(label).long()        
        

