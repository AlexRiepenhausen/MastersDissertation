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
    index             = 8
    identifier        = 9


class VectorDataset(Dataset):


    def __init__(self, file_paths, label_path, category, loadertype="train"):
    
        self.category   = category 
        self.loadertype = loadertype
    
        self.file_paths = file_paths
        self.num_files  = len(file_paths)
        
        self.labels     = self.getLabelsFromFiles(label_path)  

        self.lbl_hist   = self.labelHistogram()  
        
        self.adaptor    = ColourToOwnership()
        
        self.positive_samples = 0
        self.negative_samples = 0    
        
        self.current_test_sample = 0 
        
        self.previous_positive = dict()


    def __len__(self):
        return self.num_files
               
                
        
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
     
    
    def testingSampleDraw(self):
    
        index = self.current_test_sample
           
        if self.labels[index][labelType.property_type] != 'flat':             
                 
            index = random.randint(0, self.num_files-1)  
            while self.labels[index][labelType.property_type] != 'flat':   
                index = random.randint(0, self.num_files-1) 
                
        label      = self.labels[index][self.category]   
        classlabel = self.ownership(label)   
            
        self.current_test_sample += 1   
        
        if self.current_test_sample == self.num_files:
            self.current_test_sample = 0
            
        return index, classlabel                
                            
      
                 
    def balancedSampleDraw(self):
    
        if self.positive_samples > self.negative_samples: 
            index, classlabel = self.drawNegativeSample()      
            return index, classlabel                                
        else:    
            index, classlabel = self.drawPositiveSample()  
            return index, classlabel        
          
    
    
    def drawPositiveSample(self): 
         
        index = self.drawRandomSample("ownership") 
        
        while self.labels[index][labelType.property_type] != 'flat':   
            index = self.drawRandomSample("ownership") 
                    
        self.positive_samples += 1     
        
        self.previous_positive["file_index"] = index
            
        return index, "ownership"
        
              
        
    def drawNegativeSample(self):
    
        # 50% chance of drawing previous positve sample and looking at the negatives
        previous_len = len(self.previous_positive["non_positive_samples"])
        
        if random.randint(0, 1) == 1 and previous_len > 0:
        
            self.negative_samples += 1
            return self.previous_positive["file_index"], "previous"
                
        else:
         
            index = self.drawRandomSample("no ownership") 
            
            while self.labels[index][labelType.property_type] != 'flat':   
                index = self.drawRandomSample("no ownership")
                
            self.negative_samples += 1
            
            return index, "no ownership"
    
  
    def drawRandomSample(self, classification_label):
    
        key = ""
     
        while key != classification_label:
            
            index = random.randint(0, self.num_files-1)
            label = self.labels[index][self.category]   
            key   = self.ownership(label)
                  
        return index
                      
                
            
    def loadVectors(self, index, classlabel):
        
        # get the raw vectors and the colour pair positions within the parcel
        vectors, parcel = self.getVectors(index)
                 
        # get the tags       
        tags = self.getCurrentTags(parcel)
        
        # assign ownership to tags
        tags = self.assignOwnershipToTags(index, tags)
        
        # get index denoting required ownership type specified by classlabel
        requiredLabelIndex = self.getRequiredLabelIndex(classlabel, tags)
                          
        # remove irrelevant colours from vectors          
        vectors = self.removeIrrelevantColours(requiredLabelIndex, tags, vectors)
        
        return vectors, tags[requiredLabelIndex]
        
        
        
    def getVectors(self, index):
    
        vectors    = list()
        
        vectorfile = open(self.file_paths[index], 'r', encoding='utf8')
        line       = vectorfile.readline()
              
        position   = 0    
        parcel     = list()
    
        while line:
            
            arr = np.fromstring(line, dtype=float, sep=" ")
            vectors.append(arr)
            
            if self.colourPairIdentified(arr):
            
                key = self.adaptor.getIndex(arr)
                parcel.append((position, self.adaptor.reverseDictionary[key]))
                
            position += 1
            
            line = vectorfile.readline()     
            
        return vectors, parcel  
        
        
        
    def colourPairIdentified(self, arr):     
        difference = list(set(arr) - set(self.adaptor.padding)) 
        if len(difference) == 1:
            return True
        else:
            return False
 
 
 
    def getRequiredLabelIndex(self, classlabel, tags):
          
        if classlabel == "no ownership":
            classlabel_index = random.randint(0, len(tags)-1)
            return classlabel_index
            
        if classlabel == "ownership":
            
            ownership_indices = list()           
            non_ownership_ind = list()
            
            for i in range(0, len(tags)):
                if tags[i][2] == classlabel:
                    ownership_indices.append(i)
                else:
                    non_ownership_ind.append(i)
                    
            self.previous_positive["non_positive_samples"] = non_ownership_ind
            
            classlabel_index = random.randint(0, len(ownership_indices)-1)
            
            return ownership_indices[classlabel_index]        
            
        if classlabel == "previous":
            classlabel_index = random.randint(0, len(self.previous_positive["non_positive_samples"])-1)                 
            return self.previous_positive["non_positive_samples"][classlabel_index]     
                
        # if classlabel does not exist, return 0 as default
        return 0    
 
 
 
    def getCurrentTags(self, parcel):
    
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
            
        return tags   
 
 
 
    def assignOwnershipToTags(self, index, tags):
    
        labels      = self.labels[index][self.category]
        label_array = labels.split(", ")
        
        for i in range(0, len(tags)):
            if tags[i][0] not in label_array:
                tags[i].append("no ownership")
            else:
                tags[i].append("ownership")  
                
        return tags
            
    
    
    def removeIrrelevantColours(self, requiredLabelIndex, tags, vectors):
    
        for tag in tags:
            if tag[0] != tags[requiredLabelIndex][0]:
                positions = tag[1]
                for position in positions:
                    vectors[position] = [] 
        
        vectors = [x for x in vectors if x != []] 
        
        return vectors       
       
                
        
    def labelConversion(self, label):

        if label == "ownership":
            return 1
        else:
            return 0
            
            
            
    def debug(self, index, classlabel, label):
        print("Index: {} Class: {} Label: {}".format(index, classlabel, label))
        print("Pos:   {} Neg:   {}".format(self.positive_samples, self.negative_samples))  
        print("{} is {}".format(label[0], self.labelConversion(label[2])))
        print("------------------------------------------------")
                       
                
                
    def __getitem__(self, idx):
    
        index = 0
        classlabel = "ownership"

        if self.loadertype == "train":
            index, classlabel = self.balancedSampleDraw()    
        else:
            index, classlabel = self.testingSampleDraw()
        
                                      
        vectors, label = self.loadVectors(index, classlabel)
    
        #self.debug(index, classlabel, label)
    
        colour_pair = label[0]
        numeric_lbl = self.labelConversion(label[2])

        return torch.tensor(np.asarray(vectors)).float(), torch.tensor(numeric_lbl).long(), colour_pair  

