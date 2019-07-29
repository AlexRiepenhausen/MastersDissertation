import torch
import time
import re
import ndjson
import jsonlines
import random
from utilities.utilities import readVectorsDict, generateFilePaths, getTextNdJson
from tkinter import *
from collections import defaultdict
from utilities.utilities import labelType

class Duplicate():

    def __init__(self, filename, duplicate_factor):
    
        self.filename         = filename
        self.data             = self.readFromFile()   
         
        self.duplicate_factor = duplicate_factor
        self.stylekeys        = self.getKeywords(1, 11)
        self.colourkeys       = self.getKeywords(12,25)
        self.shuffled_colours = self.getDuplicationConfig(self.colourkeys)
        self.shuffled_styles  = self.getDuplicationConfig(self.stylekeys)
     
    
    def readFromFile(self):
        with open(self.filename) as f:  
            return ndjson.load(f)  
            
            
    def getKeywords(self, start, end):
    
        path = './data/w2v/training/dictionary/keywords.txt'
        keywords = dict()
        enum = 1
        for line in open(path, encoding="utf8"):
            if enum >= start:
                keywords[line.replace('\n','')] = enum-start
            if enum >= end:
                break
            enum += 1
            
        return keywords
        
        
    def getDuplicationConfig(self, dictionary):
    
        container    = dict()
        new_dict     = dictionary
        container[0] = {v: k for k, v in dictionary.items()}
    
        for i in range(1, self.duplicate_factor):
            new_dict = self.shuffleDictionary(new_dict)
            container[i] = new_dict
            new_dict = {v: k for k, v in new_dict.items()} # invert
            
        return container
        
        
    def shuffleDictionary(self, old_dict):
    
        assigned_numbers = list()
        new_dict         = dict()
    
        for item in old_dict:
            assigned = False
            while not assigned:
                number = random.randint(0, len(old_dict)-1)
                
                if len(assigned_numbers) < len(old_dict)-3:                  
                    if number != old_dict[item] and number not in assigned_numbers:
                        assigned_numbers.append(number)
                        new_dict[number] = item
                        assigned = True
                else:
                    if number not in assigned_numbers:
                        assigned_numbers.append(number)
                        new_dict[number] = item
                        assigned = True                        
                    
        return new_dict  
        
    
    def returnEquivalentColour(self, dict_index, word):
        wid = self.colourkeys[word] 
        return self.shuffled_colours[dict_index][wid]
        
        
    def returnEquivalentStyle(self, dict_index, word):
        wid = self.stylekeys[word] 
        return self.shuffled_styles[dict_index][wid]
        
    
    def preSelectDocuments(self, num_files, starting_point, labelSelection=None):
    
        selected = list()    

        if labelSelection == None:
            for i in range(starting_point, starting_point + num_files):
                selected.append(i) 
    
        if labelSelection == labelType.exclusive_strata:        
            for i in range(starting_point, starting_point + num_files):
                if self.data[i]['property_type'] == 'flat':
                    if self.data[i]['exclusive_strata'] != 'verbal':
                        selected.append(i) 
            
        print("Total of {} documents selected out of {}, which is {} %".format(len(selected), num_files, float(len(selected)*100/num_files)))
        print("Number of files to be converted: {} times {} = {}".format(len(selected), self.duplicate_factor, len(selected)*self.duplicate_factor))
        
        return selected
        
        
    def returnEquivalentLabels(self, dict_index, labelcontainer):
    
        lbl_arr = list()
        new_str = ''
        
        labels  = labelcontainer.split(', ')
        
        for label in labels:
            if label != 'cross hatched':
                lbl_arr.append(label.split(' '))
            else:
                lbl_arr.append(label)
              
        for label in lbl_arr:     
              
            for sub in label:
            
                if sub in self.colourkeys:                                                                                                          
                    sub = self.returnEquivalentColour(dict_index,sub)
                
                if sub in self.stylekeys:
                    sub = self.returnEquivalentStyle(dict_index, sub)                
                                
                new_str = new_str + sub + ' '
                
            new_str = new_str[:-1]
            new_str = new_str + ', '
            
        new_str = new_str[:-2]
        
        return new_str
        
        
    def convert(self, num_files, dict_file, filepath, labelpath, starting_point, labelSelection=None):
    
        ''' 
        property_type       = 0
        tenement_steading   = 1 
        exclusive_strata    = 2                       
        exclusive_solum     = 3
        common_strata       = 4
        common_solum        = 5
        additional_info     = 6
        char_count          = 7 
        index               = 8
        id                  = 9
        ''' 
    
        num_unknown_words   = 0
        total_num_words     = 0
        vector_dict, params = readVectorsDict(dict_file)

        selected = self.preSelectDocuments(num_files, starting_point, labelSelection)
        
        actual_file_num = len(selected) * self.duplicate_factor
        
        vec_files = generateFilePaths(filepath,  actual_file_num, '.txt')
        labels    = generateFilePaths(labelpath, actual_file_num, '.txt')
        
        file_index = 0
        
        for i in range(0, self.duplicate_factor): # duplication factor
        
            for j in selected:                    # number of original training files            
                
                text  = getTextNdJson(self.data, j)
                
                with open(vec_files[file_index], 'w') as f:   
                                              
                    for word in text: 
                 
                        if word in vector_dict:   # check if word exists in your dictionary
                            
                            if word in self.colourkeys:                                                                                                          
                                word = self.returnEquivalentColour(i,word)
                            
                            if word in self.stylekeys:
                                word = self.returnEquivalentStyle(i, word)
                            
                            f.write(str(vector_dict[word]).replace("'", "").replace(", ", " ").replace("[", "").replace("]", "")+'\n')                               
                                
                        else:
                            num_unknown_words += 1
                            
                        total_num_words += 1                
                
                with open(labels[file_index], 'w') as f: 
                
                
                    property_type     = self.data[j]['property_type'] 
                    
                    tenement_steading = self.returnEquivalentLabels(i, self.data[j]['tenement_steading'])    
                    exclusive_strata  = self.returnEquivalentLabels(i, self.data[j]['exclusive_strata'])                  
                    exclusive_solum   = self.returnEquivalentLabels(i, self.data[j]['exclusive_solum'])  
                    common_strata     = self.returnEquivalentLabels(i, self.data[j]['common_strata'])  
                    common_solum      = self.returnEquivalentLabels(i, self.data[j]['common_solum'])                                         
                    
                    additional_info   = self.data[j]['additional_info'] 
                    char_count        = self.data[j]['char_count'] 
                    index             = self.data[j]['index'] 
                    id                = self.data[j]['id'] 
    
                    
                    f.write(property_type     + '\n')  
                           
                    f.write(tenement_steading + '\n') 
                    f.write(exclusive_strata  + '\n')
                    f.write(exclusive_solum   + '\n')
                    f.write(common_strata     + '\n')
                    f.write(common_solum      + '\n')
                    
                    f.write(additional_info   + '\n')
                    f.write(str(char_count)   + '\n') 
                    f.write(str(index)        + '\n')
                    f.write(id                + '\n')                     
                    
                file_index = file_index + 1 
                    
                    
        percent_unknown_words = num_unknown_words*100/total_num_words
        sys.stderr.flush()
        print("\nTotal number of words {}, unknown words {}, percentage unknown words {}".format(total_num_words,
                                                                                                  num_unknown_words,
                                                                                                  percent_unknown_words))                      
               
               
     