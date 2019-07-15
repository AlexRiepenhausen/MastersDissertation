import numpy as np
import torch
import time
import random
import ndjson
import jsonlines
from torch.utils.data import Dataset
from utilities import utilities

np.random.seed(12345)

class DataReader:
    NEGATIVE_TABLE_SIZE = 1e5

    def __init__(self, primary_files, min_count, supporting_files=None):

        self.negatives = []
        self.discards  = []
        self.negpos    = 0

        self.word2id        = dict()
        self.id2word        = dict()
        self.word_frequency = dict()

        self.token_count        = 0
        self.max_num_words_file = 0

        self.primary_files      = primary_files
        self.supporting_files   = supporting_files

        self.file_paths = primary_files if supporting_files is None else primary_files + supporting_files

        self.data = dict()

        if self.ndJson:
            for item in self.primary_files:              
                with open(item) as f:
                    self.data[item] = ndjson.load(f)
            self.readWordsNdJson(min_count)      
        else:
            self.readWords(min_count)
            
        self.initTableNegatives()
        self.initTableDiscards()


    def ndJson(self, primary_files):
        if '.ndjson' in primary_files[0]:
            return True
        else:
            return False
            
    
    def getTextNdJson(self, item, index):
    
        address    = utilities.parseLine(self.data[item][index]['address'][0]['prettyPrint']) 
        text_array = utilities.parseLine(self.data[item][index]['text']).replace(address, 'address').split(' ') 

        result = list()
        
        for text in text_array:
            if len(text) > 0:
                result.append(text)
                
        self.data[item][index]['text_array'] = result  


    # read words and create word2id and id2word lookup tables
    def readWordsNdJson(self, min_count):

        print("Setting up word2vec training")
        word_frequency = dict()
        
        for item in self.primary_files:
            for i in range(0, 5000):
    
                word_count = 0            
                self.getTextNdJson(item, i)
                
                for word in self.data[item][i]['text_array']:
                    word_count += 1
                    self.token_count += 1
                    word_frequency[word] = word_frequency.get(word, 0) + 1

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        print("Read " + str(self.token_count) + " words.\n")
        print("Total embeddings: " + str(len(self.word2id))+ '\n')


    # read words and create word2id and id2word lookup tables
    def readWords(self, min_count):
        print("Setting up word2vec training")
        word_frequency = dict()
        for file in self.file_paths:
            word_count = 0
            for line in open(file, encoding="utf8"):
                line = utilities.parseLine(line).split()
                if len(line) > 1:
                    for word in line:
                        word_count += 1
                        if len(word) > 0:
                            self.token_count += 1
                            word_frequency[word] = word_frequency.get(word, 0) + 1

            if word_count > self.max_num_words_file and file in self.primary_files :
                self.max_num_words_file = word_count

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        print("Read " + str(self.token_count) + " words.\n")
        print("Total embeddings: " + str(len(self.word2id))+ '\n')


    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)


    # unigram distribution?
    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response

# -----------------------------------------------------------------------------------------------------------------

class Word2vecDataset(Dataset):
    def __init__(self, data, window_size, custom_files=None):
        self.data        = data
        self.window_size = window_size
        self.files       = data.file_paths if custom_files is None else custom_files
        self.num_files   = 5000

    def __len__(self):
        return 5000 #numdocuments

    def __getitem__(self, idx):

        findex     = random.randint(0, self.num_files-1)
        item_index = random.randint(0, len(self.data.primary_files)-1)
        item       = self.data.primary_files[item_index]
        
        if self.data.ndJson:              
            words = self.data.data[item][findex]['text_array'] 
            if len(words) > 1:   
            
                word_ids = [self.data.word2id[w] for w in words if
                            w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]
  
                boundary = np.random.randint(1, self.window_size)
  
                return [(u, v, self.data.getNegatives(v, 5)) for i, u in enumerate(word_ids) for j, v in
                        enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v]            
        else:    
            file = open(self.files[findex], 'r', encoding='utf8')

            while True:
                line = file.readline()
                if not line:
                    self.file.seek(0, 0)
                    line = file.readline()
      
                if len(line) > 1:
                    words = utilities.parseLine(line).split()
      
                    if len(words) > 1:
                        word_ids = [self.data.word2id[w] for w in words if
                                    w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]
      
                        boundary = np.random.randint(1, self.window_size)
      
                        return [(u, v, self.data.getNegatives(v, 5)) for i, u in enumerate(word_ids) for j, v in
                                enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v]

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
