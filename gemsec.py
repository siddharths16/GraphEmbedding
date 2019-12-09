import zipfile
import collections
import numpy as np
import math
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as Func
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from tqdm import tqdm
from collections import Counter
from skip_gram_model import skipgram
from GEUtils.GEUtils import *
from calculation_helper import SecondOrderRandomWalker

class GEMSEC:
    def __init__(self, graph, random_walks, degrees, vocabulary_size, embedding_dim=16, random_walk_length=80, num_of_walks=5, window_size=5, neg_sample_num=10):
        #self.op = Options(inputfile, vocabulary_size)
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.vocabulary_size = vocabulary_size
        self.num_of_walks = num_of_walks
        self.neg_sample_num = neg_sample_num
        self.random_walk_length = random_walk_length
        self.random_walks = random_walks
        self.degrees = degrees  #maybe compute degrees using Counter(walks[0]).most_common(vocab_size-1)
        deg = self.count_frequency_values()
        self.sample_table = self.init_sample_table(deg)

    def train(self):
        print("Vocabulary size: ", self.vocabulary_size)
        model = skipgram(self.vocabulary_size, self.embedding_dim)

        if torch.cuda.is_available():
            print("Using GPU")
            model.cuda()
            
        self.print_model_parameters(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3) ##not used currently
        
        self.current_step = 0
        
        for epoch in range(self.num_of_walks):
            print("Epoch: ", epoch)
            random.shuffle(self.nodes)

            for repetition in tqdm(self.nodes):
                self.current_step = self.current_step + 1
                
                # your code here    
                """Mycode"""
                batch_inputs = batch_input_generator(self.random_walks[self.current_step-1], self.random_walk_length, self.window_size) ## center word
                batch_labels = batch_label_generator(self.random_walks[self.current_step-1], self.random_walk_length, self.window_size) ## context word
                batch_size = len(batch_inputs)

                pos_u, pos_v, neg_v = self.new_generate_batch(batch_inputs, batch_labels, self.window_size, self.neg_sample_num, self.random_walks[self.current_step-1])
                """ends"""
                
                pos_u = Variable(torch.LongTensor(pos_u))
                pos_v = Variable(torch.LongTensor(pos_v))
                neg_v = Variable(torch.LongTensor(neg_v))
                
                if torch.cuda.is_available():
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()
                                
                optimizer.zero_grad()
                loss = model(pos_u, pos_v, neg_v,batch_size)
                    
                loss.backward()
   
                optimizer.step()

            #modularity, assignments = neural_modularity_calculator(self.graph, model.input_embeddings() , model.get_cluster_means())
            modularity, assignments = classical_modularity_calculator(self.graph, model.input_embeddings())
            print("loss: ", loss, " Modularity: ", modularity, " Learning rate: ", self.get_lr(optimizer) )       
            scheduler.step(loss)
            print()
        print("Optimization Finished!")

    def new_generate_batch(self, batch_inp, batch_lb, window_size, neg_samp_num, curr_walk):
        pos_u = []
        pos_v = []
        batch_size = len(batch_inp)
        
        for i in range(batch_size):
                
            for j in range(window_size):
                pos_u.append(batch_inp[i])     
                pos_v.append(batch_lb[i,j]) 
        
        neg_v = np.random.choice(self.sample_table, size=(batch_size*window_size, neg_samp_num))
        return pos_u, pos_v, neg_v
    
    def init_sample_table(self, deg):
        count = [ele[1] for ele in deg]
        pow_frequency = np.array(count)**0.75
        power = sum(pow_frequency)
        ratio = pow_frequency/ power
        table_size = 1e8
        count = np.round(ratio*table_size)
        sample_table = []
        for idx, x in enumerate(count):
            sample_table += [idx]*int(x)
        return np.array(sample_table)
    
    def count_frequency_values(self):
        raw_counts = [node for walk in self.random_walks for node in walk]
        counts = Counter(raw_counts).most_common()
        return counts
    
    def print_model_parameters(self,model):
        print(model)
        for name, param in model.named_parameters():
            print(name, param.size(), param.requires_grad)
        
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            
            
if __name__ == '__main__':
    
    graph = read_graph('./data/artist_edges.csv', 'csv')
    p=1
    q=1
    window_size = 10 
    num_of_walks = 20 ##epochs
    random_walk_length = 80
    walker = SecondOrderRandomWalker(graph, False, p, q)
    
    walker.preprocess_transition_probs()
    print("Transition proabability computed")
    walks, degrees = walker.simulate_walks(num_of_walks, random_walk_length)
    vocab_size =  len(degrees)
    
    gemsec_model = GEMSEC(graph=graph, random_walks=walks, degrees=degrees, vocabulary_size=vocab_size, 
                          embedding_dim=16, random_walk_length=random_walk_length, num_of_walks=num_of_walks,
                          window_size=window_size, neg_sample_num=10)

    gemsec_model.train()



