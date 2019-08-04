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
        
        self.ownershipflag = True



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
               
        while self.labels[index][labelType.property_type] != 'flat': 
            index += 1
            self.current_test_sample += 1
                                                       
        vectors, parcel    = self.getVectors(index)                  
        tags               = self.getCurrentTags(parcel)                   
        tags               = self.assignOwnershipToTags(index, tags)  
        classlabel         = self.decideLabel(tags)
        requiredLabelIndex = self.getLabelIndexTest(tags, classlabel)
        vectors            = self.truncateVectors(requiredLabelIndex, tags, vectors)
        
        self.current_test_sample += 1 
        
        if self.current_test_sample == self.num_files:
            self.current_test_sample = 0
        
        return index, vectors, tags[requiredLabelIndex]     
        
     
    
    def getLabelIndexTest(self, tags, classlabel):
            
        indices = list()           
        
        for i in range(0, len(tags)):
            if tags[i][2] == classlabel:
                indices.append(i)
        
        classlabel_index = random.randint(0, len(indices)-1)
        
        if classlabel == "ownership":
            self.positive_samples += 1
            
        if classlabel == "no ownership":
            self.negative_samples += 1
        
        return indices[classlabel_index]        
         
     
     
    def decideLabel(self, tags):
    
        both         = False
        ownership    = False
        no_ownership = False
        
        for tag in tags:
            if tag[2] == "ownership":
                ownership = True
            if tag[2] == "no ownership":
                no_ownership = True
                
        if ownership == True and no_ownership == True:
            both = True                                            

        if both:
            if self.positive_samples > self.negative_samples:
                return "no ownership"
            else:
                return "ownership"
        
        if ownership:
            return "ownership"
            
        if no_ownership:
            return "no ownership"   
     
     
 
    def getOwnershipIndices(self, tags, classlabel):
        ownership_ind = list()
        for i in range(0, len(tags)):
            if tags[i][2] == classlabel:
                ownership_ind.append(i)  
        return ownership_ind                                                                                                         
      
      
                 
    def balancedSampleDraw(self):
        
        if self.positive_samples > self.negative_samples: 
            self.ownershipflag = False    
            
        if self.negative_samples > self.positive_samples: 
            self.ownershipflag = True      
    
        if self.ownershipflag == True:
            index, classlabel = self.drawPositiveSample()  
            return index, classlabel 
            
        if self.ownershipflag == False:
            index, classlabel = self.drawNegativeSample()      
            return index, classlabel  
            
                                                          
    
    def drawPositiveSample(self): 
         
        index = self.drawRandomSample("ownership") 
        
        while self.labels[index][labelType.property_type] != 'flat':   
            index = self.drawRandomSample("ownership") 
                    
        self.positive_samples += 1     
            
        return index, "ownership"
        
              
        
    def drawNegativeSample(self):
        
        while True:
     
            non_ownership_ind = list()
            
            # get the raw vectors and the colour pair positions within the parcel
            index             = self.drawRandomSample("ownership")
            vectors, parcel   = self.getVectors(index)                  
            tags              = self.getCurrentTags(parcel)                   
            tags              = self.assignOwnershipToTags(index, tags)
                        
            for i in range(0, len(tags)):
                if tags[i][2] == "no ownership":
                    non_ownership_ind.append(i)
                    
            if len(non_ownership_ind) > 0:
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
        requiredLabelIndex = self.getRequiredLabelIndex(index, classlabel, tags)
                          
        # remove irrelevant colours from vectors          
        vectors = self.truncateVectors(requiredLabelIndex, tags, vectors)
        
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
 
   
 
    def getRequiredLabelIndex(self, index, classlabel, tags):
            
        ownership_indices = list()           
        
        for i in range(0, len(tags)):
            if tags[i][2] == classlabel:
                ownership_indices.append(i)
            
        classlabel_index = random.randint(0, len(ownership_indices)-1)
        
        return ownership_indices[classlabel_index]        

 
 
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
            
    
    
    def truncateVectors(self, requiredLabelIndex, tags, vectors):
        
        cut_off_point = 0
        
        for tag in tags:
            if tag[0] == tags[requiredLabelIndex][0]:
                positions     = tag[1]
                cut_off_point = positions[1] + 1
                break
                
        return vectors[0:cut_off_point]       
       
                
        
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
            vectors, label    = self.loadVectors(index, classlabel)
            colour_pair       = label[0]
            numeric_lbl       = self.labelConversion(label[2]) 
            doc_id            = self.labels[index][8] 
            doc_index         = self.labels[index][9]   
            return torch.tensor(np.asarray(vectors)).float(), torch.tensor(numeric_lbl).long(), colour_pair, doc_id, doc_index 
        else:
            index, vectors, label = self.testingSampleDraw()    
            colour_pair    = label[0]
            numeric_lbl    = self.labelConversion(label[2])      
            doc_id         = self.labels[index][8] 
            doc_index      = self.labels[index][9]                   
            return torch.tensor(np.asarray(vectors)).float(), torch.tensor(numeric_lbl).long(), colour_pair, doc_id, doc_index  

