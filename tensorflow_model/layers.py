# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:09:37 2019

@author: siddh
"""

import math 
import numpy as np 
import torch 
import torch.distributions as tdist
import torch.nn as nn


class DeepWalker:
    """
    DeepWalk embedding layer class
    """
    
    def __init__(self, args, vocab_size, degrees):
        self.args = args
        self.vocab_size = vocab_size
        self.degrees = degrees
        self.train_labels = torch.tensor((), dtype=torch.int64)
        self.embedding_matrix = torch.FloatTensor(self.vocab_size, 
                                               self.args.dimensions).uniform_(-0.1/self.args.dimensions, 0.1/self.args.dimensions)
        self.nce_weights = torch.FloatTensor(self.vocab_size, self.args.dimensions).uniform_(std=1.0/np.sqrt(self.args.dimensions))
        self.nce_biases = torch.FloatTensor(self.vocab_size).uniform_(-0.1/self.args.dimensions, 0.1/self.args.dimensions)
        
        
    def __call__(self):
        """
        Calculating the embedding cost with NCE 
        """
        self.train_labels_flat = self.train_labels.view(-1,1)
        self.input_ones = torch.ones(self.train_labels.shape)
        self.train_inputs_flat = torch.mul(self.input_ones, self.train_inputs.view(-1,1)).view(-1)
        
        embedding_lookup = nn.Embedding.from_pretrained(self.embedding_matrix)
        embedding_lookup.weight.requires_grad = False
        self.embedding_partial = embedding_lookup(self.train_inputs_flat)
        
class Clustering:
    
    def __init__(self, args):
        """
        Initializing the cluster center matrix
        """
        self.args = args
        self.cluster_means = torch.FloatTensor(self.args.cluster_number, 
                                               self.args.dimensions).uniform_(-0.1/self.args.dimensions, 0.1/self.args.dimensions)
        
    def __call__(self, Walker):
        """
        Calculating the clustering cost
        """
        
        self.clustering_differences = Walker.embedding_partial.unsqueeze(1) - self.cluster_means ##Add a dimension at axis-1
        self.cluster_distances = self.clustering_differences.norm(p='fro', dim=2)
        self.to_be_averaged = torch.min(self.cluster_distances, axis=1)
        return torch.mean(self.to_be_averaged)


class Regularization:
    """
    Smooth Regularization class
    """
    
    def __init__(self, args):
        self.args = args
        self.edge_indices_right = torch.tensor((), dtype=torch.int64)
        self.edge_indices_left = torch.tensor((), dtype=torch.int64)
        self.overlap = torch.tensor((), dtype=torch.float32)
    
    def __call__(self, Walker):
        """
        Calculating the reugalization cost.
        """
        #https://stackoverflow.com/questions/55133931/how-to-transfer-the-follow-embedding-code-in-tensorflow-to-pytorch
        embedding_left = nn.Embedding.from_pretrained(Walker.embedding_partial)
        embedding_left.weight.requires_grad = False
        self.left_features = embedding_left(self.edge_indices_left)
        
        embedding_right = nn.Embedding.from_pretrained(Walker.embedding_partial)
        embedding_right.weight.requires_grad = False
        self.left_features = embedding_right(self.edge_indices_right)
        
        self.regularization_differences = self.left_features-self.right_features
        noise =  np.random.uniform(-self.args.regularization_noise,
                                   self.args.regularization_noise,
                                   (self.args.random_walk_length-1, self.args.dimensions))
        
        self.regularization_differences = self.regularization_differences + noise
        
        self.regularization_distances  = self.regularization_differences.norm(p='fro', dim=1)
        self.regularization_distances = self.regularization_distances.view([-1,1])
        self.regularization_loss = torch.mean(torch.matmul(self.overlap.t(), self.regularization_distances))
        return self.args.lambd*self.regularization_loss
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    