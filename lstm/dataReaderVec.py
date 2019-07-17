import torch
import numpy as np
import random
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict

class VectorDataset(Dataset):

    def __init__(self, file_paths, labels, seq_dim, batch_size=1):
    
        self.file_paths = file_paths
        self.num_files  = len(file_paths)
        self.batch_size = batch_size
        
        self.keywords   = self.getKeywords()
        self.labels     = self.getLabelsFromFiles(labels)    
        self.files      = self.readFiles()    

        # print label distribution
        self.lbl_hist   = self.labelHistogram()


    def __len__(self):
        return self.num_files/self.batch_size
        
  
    def getKeywords(self):
        path = './data/w2v/training/dictionary/labels.txt'
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
        
    
    def printLabelDictionary(self, lbl_hist):
    
        print("| ---- Label Dictionary ----  |")
        print("|                             |")
        for item in lbl_hist:
            print("| Label: {:2d}, Description: {} ".format(item, self.labels[item]))
        print("|                             |")      


    def labelHistogram(self):
        lbl_hist = defaultdict(int)
        for label in self.labels:
            lbl = self.keywords[label[2]]
            lbl_hist[lbl] += 1

        lbl_hist = OrderedDict(sorted(lbl_hist.items()))
        
        print("| ---- Label Frequency ----  |")
        print("|                            |")
        for item in lbl_hist:
                print("| Label: {:2d}, Frequency: {:4d} |".format(item, lbl_hist[item]))
        print("|                            |")
        
        self.printLabelDictionary(lbl_hist) 
        
        return lbl_hist    
        
        
    def drawSample(self):
    
        threshold  = random.uniform(0, 0.35)
    
        while True:
            index      = random.randint(0, self.num_files-1)
            label_str  = self.labels[index][2]
            label      = self.keywords[label_str]
            
            draw_prob  = 1.0 - float(self.lbl_hist[label]/self.num_files)
            
            if draw_prob > threshold:
                return index
                


    def __getitem__(self, idx):
    
        index      = self.drawSample()
        
        vectorfile = open(self.file_paths[index], 'r', encoding='utf8')
    
        vectors = list()
        line    = vectorfile.readline()
  
        while line:
            arr = np.fromstring(line, dtype=float, sep=" ")
            vectors.append(arr)
            line = vectorfile.readline()  
            
        label_str  = self.labels[index][2]
        
        '''
        index      = self.drawSample()
        vectorfile = self.files[index]
        label_str  = self.labels[index][2]
        label      = self.keywords[label_str]
        '''
        
        label = 0
        if label_str == 'verbal':
            print('verbal')
            label = 1
        else:
            print('non-verbal')
        
        return torch.tensor([np.asarray(vectors)]).float(), torch.tensor(label).long()

