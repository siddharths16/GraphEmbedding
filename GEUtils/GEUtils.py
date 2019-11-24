# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:54:35 2019

@author: siddh
"""
import time
import random
import numpy as np
import networkx as nx 
from tqdm import tqdm
import numpy.random as npr
    
def read_graph(gFile, g_format):
    '''
    Reads the input graph in networkx
    gFile: Path to the graph
    g_format: Format of the graph. Check networkx graph types.
    '''
    
    if (g_format == "edgelist"):
        Graph = nx.read_weighted_edgelist(gFile, nodetype=int)
        return Graph
    else: 
        print("%s Format not supported" %format)
        return
        
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
    
