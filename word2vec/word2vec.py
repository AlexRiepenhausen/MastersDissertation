import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""

class SkipGramModel(nn.Module):

    def __init__(self, keyword_path, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.keywords      = self.readKeywords(keyword_path)
        self.num_keywords  = len(self.keywords)
        self.emb_size      = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings  = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings  = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.weight_init()


    # run vector through word2vec shallow neural network
    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)


    # save the hidden layer values for every vector
    def save_embedding(self, id2word, file_name, max_num_words_file):
    
        embedding = self.u_embeddings.weight.cpu().data.numpy() 
        padding   = np.zeros((self.num_keywords,), dtype=float)  
        
        with open(file_name, 'w') as f:
        
            f.write('%d %d %d\n' % (len(id2word), self.emb_dimension + self.num_keywords, max_num_words_file))
    
            for wid, w in id2word.items():       
            
                e = None    
                   
                if w in self.keywords:
                    key_vec        = np.zeros((self.emb_dimension + self.num_keywords,), dtype=float)   
                    index          = self.emb_dimension + self.keywords[w]
                    key_vec[index] = 1.0
                    e = ' '.join(map(lambda x: str(x), key_vec))
                else:  
                    word_vec = np.concatenate((embedding[wid], padding), axis=None)
                    e        = ' '.join(map(lambda x: str(x), word_vec))
                                        
                f.write('%s %s\n' % (w, e))     


    # initialise (or refresh) weights
    def weight_init(self):
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)
        
        
    def readKeywords(self, keywords_file):
        keywords = dict()
        index = 0
        for line in open(keywords_file, encoding="utf8"):
            keywords[line.replace('\n','')] = index #0,1,2,3 ...
            index += 1
        return keywords

