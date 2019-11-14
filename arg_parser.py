#!/usr/bin/env python
# coding: utf-8

# In[11]:


import argparse


# In[38]:


def arg_parser():
    
    '''
    For parsing the command line arguments.
    '''
    parser = argparse.ArgumentParser(description='Run GraphEmbedding')
    
    parser.add_argument('--graph', default='graph/karate.edgelist', nargs='?', type=str,                         required=True, help='Path to input graph')
    
    parser.add_argument('--dimensions', default=10, type=int,                 required=True, help='Number of embedding dimensions. Default is 10')
    
    parser.add_argument('--clusters', default=2, type=int,                         required=True, help='Number of clusters. Default is 2')
    
    parser.add_argument('--walk-length', default=50, type=int,                          help='length of sequences i.e. random walk length for every node.')

    parser.add_argument('--window-size', default=5, type=int,                          help='Window size')
    
    parser.add_argument('--gamma', default=0.1, type=float,                          help='Initial clustering weight coefficient. Default is 0.1')
    
    parser.add_argument('--initial-learning-rate', default=0.01, type=float,                          help='Initial learning rate. Default is 0.01')
    
    parser.add_argument('--final-learning-rate', default=0.001, type=float,                          help='Final learning rate. Default is 0.001')
    
    parser.add_argument("--annealing-factor", type=float, default=1,                          help="Annealing factor. Default is 1.0.")
    
    return parser


# In[41]:


#parser = arg_parser()
#parser.parse_args(['--graph', 'mypath'])


# In[ ]:




