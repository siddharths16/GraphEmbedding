# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:54:35 2019

@author: siddh
"""
import math
import time
import random
import numpy as np
import pandas as pd
import networkx as nx 
import numpy.random as npr
import community
from sklearn.cluster import KMeans

def neural_modularity_calculator(graph, embedding, means):
    """
    Function to calculate the GEMSEC cluster assignments.
    """
    assignments = {}
    for node in graph.nodes():
        positions = means-embedding[node, :]
        values = np.sum(np.square(positions), axis=1)
        index = np.argmin(values)
        assignments[int(node)] = int(index)
    modularity = community.modularity(assignments, graph)
    return modularity, assignments


def classical_modularity_calculator(graph, embedding, cluster_number=20):
    """
    Function to calculate the DeepWalk cluster centers and assignments.
    """
    kmeans = KMeans(n_clusters=cluster_number, random_state=0, n_init=1).fit(embedding)
    assignments = {i: int(kmeans.labels_[i]) for i in range(0, embedding.shape[0])}
    modularity = community.modularity(assignments, graph)
    return modularity, assignments


    
def batch_input_generator(a_walk, random_walk_length, window_size):
    """
    Function to generate features from a node sequence.
    """
    seq_1 = [a_walk[j] for j in range(random_walk_length-window_size)]
    seq_2 = [a_walk[j] for j in range(window_size, random_walk_length)]
    return np.array(seq_1 + seq_2)

def batch_label_generator(a_walk, random_walk_length, window_size):
    """
    Function to generate labels from a node sequence.
    """
    grams_1 = [a_walk[j+1:j+1+window_size] for j in range(random_walk_length-window_size)]
    grams_2 = [a_walk[j-window_size:j] for j in range(window_size, random_walk_length)]
    return np.array(grams_1 + grams_2)


def read_graph(gFile, g_format):
    '''
    Reads the input graph in networkx
    gFile: Path to the graph
    g_format: Format of the graph. Check networkx graph types.
    '''
    if (g_format == "csv"):
        edges = pd.read_csv(gFile)
        Graph = nx.from_edgelist(edges.values.tolist())
        return Graph
    
    elif (g_format == "edgelist"):
        Graph = nx.read_weighted_edgelist(gFile, nodetype=int)
        return Graph
    else: 
        print("%s Format not supported" %format)
        return
        
    

def draw_graph(graph):
    print("Drawing Graph")
    nx.draw(graph, with_labels=True)


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    """
    Algorithm: Vose's Alias Method
    http://www.keithschwarz.com/darts-dice-coins/
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    smaller = []
    larger = []
    
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
            
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
            
    return J, q
            

def alias_draw(J,q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)
    
    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
    
