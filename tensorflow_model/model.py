# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:08:00 2019

@author: siddh
"""

import math
import time
import random
import numpy as np
import networkx as nx
import tqdm as tqdm
import torch
#from GEUtils.GEUtils import overlap_generator
#from GEUtils.GEUtils import neural_modularity_calculator, classical_modularity_calculator, gamma_incrementer
from random_walker import RandomWalker, SecondOrderRandomWalker
#from GEUtils.GEUtil  import index_generation, batch_input_generator, batch_label_generator


class Model(object):
    """
    Abstract model class
    """
    
    def __init__(self, args, graph):
        
        self.args = args
        self.graph = graph
        if self.args.walker == "first":
            self.walker = RandomWalker(self.graph, nx.nodes(graph),
                                       self.args.num_walks,
                                       self.args.walk-length)
            
            self.degress, self.walks =  self.walker.do_walk()
            
        else:
            self.walker = SecondOrderRandomWalker(self.graph, False, self.args.p, self.args.q)
            self.walker.preprocess_transition_probs()
            self.walks, self.degrees = self.walker.simulate_walks(self.args.num_walks, self.args.walk-length)
        
        self.nodes = self.graph.nodes()
        del self.walker
        self.vocab_size = len(self.degrees)
        self.true_step_size = self.args.num_walks*self.vocab_size
        self.build()
        
    def build(self):
        """
        Building the model
        """
        pass
    
    def feed_dict_generator(self):
        """
        Creating the feed generator
        """
        pass
    
    def train(self):
        """
        Training the model
        """
        pass
    

class GEMSECWithRegularization(Model):
    
    """
    Regularized GEMSEC class
    """
    def build(self):
        """
        Method to create the computational graph.
        """
        pass
    
    def feed_dict_generator(self, a_random_walk, step, gamma):
        """
        Method to generate:
        1. random walk features.
        2. left and right handside matrices.
        3. proper time index and overlap vector.
        """
        pass
    
   def train(self):
       """
       Method for:
       1. training the embedding.
       2. logging.
       This method is inherited by GEMSEC and DeepWalk variants without an override.
       """
        
       pass
        
   
class GEMSEC(GEMSECWithRegularization):
    """
    GEMSEC class
    """
    
    def build(self):
        """
        Method to create the computational graph.
        """
        pass
    
    def feed_dict_generator(self, a_random_walk, step, gamma):
        """
        Method to generate:
        1. random walk features.
        2. left and right handside matrices.
        3. proper time index and overlap vector.
        """
        pass
            
            
        