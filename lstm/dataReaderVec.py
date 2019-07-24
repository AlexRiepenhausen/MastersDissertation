import torch
import numpy as np
import random
from enum import IntEnum
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict
from lstm.adaptor import ColourToOwnership

class labelType(IntEnum):
    property_type     = 0
    tenement_steading = 1 
    exclusive_strata  = 2                       
    exclusive_solum   = 3
    common_strata     = 4
    common_solum      = 5
    additional_info   = 6
    char_count        = 7  


class VectorDataset(Dataset):

    def __init__(self, file_paths, label_path, category, batch_size=1):
    
        self.category   = category 
    
        self.file_paths = file_paths
        self.num_files  = len(file_paths)
        self.batch_size = batch_size
        
        self.labels     = self.getLabelsFromFiles(label_path)  

        # print label distribution
        self.lbl_hist   = self.labelHistogram()  
        
        self.adaptor    = ColourToOwnership()
        
        self.repetition = 1
        self.rep_count  = 0
        self.index      = 0
        

    def __len__(self):
        return self.num_files/self.batch_size
               
        
    def getLabelsFromFiles(self, label_path):
    
        labels = list() #list of lists
        for file in label_path:
            lines = list()
            for line in open(file, encoding="utf8"):
                lines.append(line.replace('\n',''))    
            labels.append(lines)
    
        return labels


    def labelHistogram(self):
    
        lbl_hist = defaultdict(int)
        
        for label in self.labels:
               
            key = self.ownership(label[self.category])
            lbl_hist[key] += 1    
        
        return lbl_hist  
        
    
    def ownership(self, label):
        if label == 'none':
            return "no ownership"
        elif label == 'verbal':           
            return "no ownership"
        else: 
            return "ownership" 
        
  
    def drawSample(self):
    
        prob0 = float(self.lbl_hist["no ownership"]/self.num_files) # high  probability
        prob1 = float(self.lbl_hist["ownership"]/self.num_files)    # lower probability
        
        lowth     = prob0*prob0/prob1        
        
        draw_prob = random.uniform(0.0, lowth)
            
        while True:
            
            index = random.randint(0, self.num_files-1)
            label = self.labels[index][self.category]   
            key   = self.ownership(label)
            
            if key == "no ownership":
                if draw_prob > prob0: 
                    return index   
            else:
                return index
            
            
    def loadVectors(self):
        
        vectors    = list()
        
        vectorfile = open(self.file_paths[self.index], 'r', encoding='utf8')
        line       = vectorfile.readline()
        labels     = self.labels[self.index][self.category]   
              
        position   = 0    
        parcel     = list()
        
        while line:
            
            arr = np.fromstring(line, dtype=float, sep=" ")
            vectors.append(arr)
            
            difference = list(set(arr) - set(self.adaptor.padding))
            
            if len(difference) == 1:
            
                key = self.adaptor.getIndex(arr)
                parcel.append((position, self.adaptor.reverseDictionary[key]))
                
            position += 1
            
            line = vectorfile.readline()   
                
        tags     = list()
        prev_pos = parcel[0][0]
        prev_key = parcel[0][1]
        
        for i in range(1,len(parcel)):
        
            pos = parcel[i][0]
            key = parcel[i][1]
            
            if pos == prev_pos + 1:
                tag = prev_key + " " + key
                tags.append([tag,[prev_pos, pos]])
                
            prev_pos = parcel[i][0]
            prev_key = parcel[i][1]
        
        # assign the labels
        label_array     = labels.split(", ")
        self.repetition = len(tags)
        
        for i in range(0, len(tags)):
            if tags[i][0] not in label_array:
                tags[i].append("no ownership")
            else:
                tags[i].append("ownership") 
                          
        # remove irrelevant colours []    
        for tag in tags:
            if tag[0] != tags[self.rep_count][0]:
                positions = tag[1]
                for position in positions:
                    vectors[position] = [] 
        
        vectors = [x for x in vectors if x != []]
        
        return vectors, tags[self.rep_count][2]
       
        
    def labelConversion(self, label):

        if label == "ownership":
            return 1
        else:
            return 0
            
            
    def debug(self, label, length):
        print("Index: {} Vectorlen: {}".format(self.index, length))
        print("Count: {} RepLength: {}".format(self.rep_count, self.repetition))
        print("Label: {} Numerical: {}".format(label, self.labelConversion(label)))     
                       
                
    def __getitem__(self, idx):
    
        # if memory error, load the labels as well
        if self.rep_count == self.repetition-1:     
            
            self.index = self.drawSample() 
            
            while self.labels[self.index][labelType.property_type] != 'flat':   
                self.index = self.drawSample()
            
            self.rep_count = 0     
                                                                   
        else:
            self.rep_count += 1
                    
        vectors, label = self.loadVectors()

        return torch.tensor(np.asarray(vectors)).float(), torch.tensor(self.labelConversion(label)).long()  
        

