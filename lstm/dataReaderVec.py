import torch
import numpy as np
import random
from enum import IntEnum
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict

class labelType(IntEnum):
    property_type     = 0
    tenement_steading = 1 
    exclusive_strata  = 2                       
    exclusive_solum   = 3
    common_strata     = 4
    common_solum      = 5
    additional_info   = 6
    char_count        = 7  
    index             = 8
    identifier        = 9

class VectorDataset(Dataset):

    def __init__(self, file_paths, label_path, category, loadertype="train"):
    
        self.category   = category 
        self.loadertype = loadertype    
  
        self.file_paths = file_paths
        self.num_files  = len(file_paths)
        
        self.keywords   = self.getKeywords()
        self.labels     = self.getLabelsFromFiles(label_path)    
        self.files      = self.readFiles()    

        # print label distribution
        self.lbl_hist   = self.labelHistogram()
        
        self.positive_samples = 0
        self.negative_samples = 0    


    def __len__(self):
        return self.num_files
        
  
  
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
        


    def labelHistogram(self):
        lbl_hist = defaultdict(int)
        for label in self.labels:
            lbl = self.keywords[label[2]]
            lbl_hist[lbl] += 1

        lbl_hist = OrderedDict(sorted(lbl_hist.items()))
        
        return lbl_hist    
     
             
        
    def drawBalancedSample(self):
            
        if self.positive_samples > self.negative_samples:
            return self.drawNegativeSample()  
        else:
            return self.drawPositiveSample()      
    
    
    
    def drawNegativeSample(self):
    
        index, label_str = self.drawRandomSample()

        while self.labelStringToClass(label_str) != 0:
            index, label_str = self.drawRandomSample()
        
        self.negative_samples += 1
        
        return index, 0
        
        
        
    def drawPositiveSample(self):
    
        index, label_str = self.drawRandomSample()

        while self.labelStringToClass(label_str) != 1:
            index, label_str = self.drawRandomSample()
        
        self.positive_samples += 1
        
        return index, 1   
        
        
        
    def drawRandomSample(self):
        index      = random.randint(0, self.num_files-1) 
        label_str  = self.labels[index][self.category]        
        return index, label_str    
        
        
        
    def labelStringToClass(self, label_str):
    
        if label_str == 'none':
            return 0  
        elif label_str == 'verbal':
            return 0
        else:
            return 1
            
            
    def labelClassToString(self, num_label):

        if num_label == 0:
            return "verbal_or_none"
        else:
            return "colour_pair"
            
                
    
    def getVectors(self, index):
    
        vectorfile = open(self.file_paths[index], 'r', encoding='utf8')
    
        vectors = list()
        line    = vectorfile.readline()
  
        while line:
            arr = np.fromstring(line, dtype=float, sep=" ")
            vectors.append(arr)
            line = vectorfile.readline()  
            
        return vectors
                


    def __getitem__(self, idx):
    
        index, label = self.drawBalancedSample()
        vectors      = self.getVectors(index)
        str_label    = self.labelClassToString(label)      
        doc_id       = self.labels[index][8] 
        doc_index    = self.labels[index][9]    
        
        return torch.tensor(np.asarray(vectors)).float(), torch.tensor(label).long(), str_label, doc_id, doc_index


