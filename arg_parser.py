#!/usr/bin/env python
# coding: utf-8
import argparse

def arg_parser():
    
    '''
    For parsing the command line arguments.
    '''
    parser = argparse.ArgumentParser(description='Run GraphEmbedding')
    
    parser.add_argument('--graph', default='graph/karate.edgelist', nargs='?', type=str, required=True, help='Path to input graph')
    
    parser.add_argument('--dimensions', default=16, type=int, required=True, help='Number of embedding dimensions. Default is 10')
    
    parser.add_argument('--cluster_number', default=2, type=int, required=True, help='Number of clusters. Default is 2')
    
    parser.add_argument('--walk-length', default=50, type=int, help='length of sequences i.e. random walk length for every node.')

    parser.add_argument('--window-size', default=5, type=int, help='Window size')

    parser.add_argument('--num_walks', default=10, type=int, help='Number of walks per node.')
    
    parser.add_argument("--walker", nargs="?", default="first", help="Random Walk order. First or Second. Default is first")
    
    ##p=1 & q=1 corresponds to Deep Walk
    parser.add_argument('--p', default=1, type=float, help='Return hyperparameter for controlling random walk.')
    
    parser.add_argument('--q', default=1, type=float, help='Inout hyperparameter for controlling random walk.')
    
    parser.add_argument("--regularization-noise",
                        type=float,
                        default=10**-8,
	                help="Uniform noise max and min on the feature vector distance.")
    
    parser.add_argument("--lambd",
                        type=float,
                        default=2.0**-4,
	                help="Smoothness regularization penalty. Default is 0.0625.")
    
    parser.add_argument('--gamma', default=0.1, type=float, help='Initial clustering weight coefficient. Default is 0.1')
    
    parser.add_argument('--initial-learning-rate', default=0.01, type=float, help='Initial learning rate. Default is 0.01')
    
    parser.add_argument('--final-learning-rate', default=0.001, type=float, help='Final learning rate. Default is 0.001')
    
    parser.add_argument("--annealing-factor", type=float, default=1, help="Annealing factor. Default is 1.0.")
    
    return parser




#parser.parse_args(['--graph', 'mypath'])


