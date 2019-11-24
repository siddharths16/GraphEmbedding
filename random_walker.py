# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:31:23 2019

@author: siddh
"""

from collections import Counter
import matplotlib.pyplot as plt
import GEUtils.GEUtils

class RandomWalker():
    '''
    Randomwalker class generetes the vertex sequence which are visited during the random walk
    from every node in every interation. These vertex sequences are then handed over to skip-gram 
    model to learn the graph embedding.
    '''
    def __init__(self, graph, reps, length):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.edges = graph.number_of_edges()
        self.length = length
        self.reps = reps
        self.walks = []
        print("Random Walking Initialized for Graph with nodes: ", self.nodes, \
              " and edges: ", self.edges)
        
    def first_order_random_walk(self, start_node):
        
        walk = [start_node]
        while(len(walk) != self.length):
            curr_node = walk[-1]
            neighbors = list(nx.neighbors(G=self.graph, n=curr_node))
            if (len(neighbors) > 0):
                #equal probability of choosing any neighbour i.e P(x(t+1)|x(t)) = 1/d(x(t+1))
                walk += random.sample(neighbors,1)  
            else:
                break
            
        return walk
    
    def count_frequency_values(self):
        """
        Calculate the co-occurence frequencies.
        """
        raw_counts = [node for walk in self.walks for node in walk]
        counts = Counter(raw_counts)
        self.degrees = [counts[i] for i in range(0, len(self.nodes))]
        return self.degrees
    
    def do_walk(self):
        ##Number of walks = self.reps*self.nodes
        
        for rep in range(self.reps):
            random.shuffle(self.nodes)
            print(" ")
            print("Random walk series " + str(rep+1) + ". initiated.")
            print(" ")
            for node in tqdm(self.nodes):
                walk = self.first_order_random_walk(node)
                self.walks.append(walk)
                
        self.count_frequency_values()
        return self.degrees, self.walks
            
            
            
class SecondOrderRandomWalker():
        
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.nodes = nx.nodes(self.G)
        print("Edge Weighting.\n")
        for edge in tqdm():
            self.G[edge[0]][edge[1]]["weight"] = 1.0
            self.G[edge[1]][edge[0]]["weight"] = 1.0
            
        self.is_directed = is_directed
        self.p = p
        self.q = q
        
    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        
        walk = [start_node]
        
        while len(walk) < walk_length:
            curr_node = walk[-1]
            curr_nbrs = sorted(G.neighbours(curr_node))
            if len(curr_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(curr_nbrs[alias_draw(alias_nodes[curr_node][0], alias_nodes[curr_node][1])])
                else:
                    prev = walk[-2]
                    next = curr_nbrs[alias_draw(alias_edges[(prev, curr_node)][0], alias_edges[(prev, curr_node)][1])]
                    walk.append(next)
            else:
                break
                 
        return walk
                   
    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        
        for walk_iter in range(num_walks):
            print(" ")
            print("Random Walk Series " + str(walk_iter+1) + ". initiated.")
            print(" ")
            random.shuffle(nodes)
            for node in tqdm(nodes):
                walks.append(self.node2vec_walk(walk_length, node))
        
        return walks, self.count_frequency_values(walks)
    

    def count_frequency_values(self, walks):
        """
        Calculate the co-occurence frequencies
        """
        raw_counts = [node for walk in walks for node in walk]
        counts = Counter(raw_counts)
        self.degrees = [counts[i] for i in range(0,len(self.nodes))]
        return self.degrees
        
    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q
        
        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src: #dtx = 0
                unnormalized_probs.append(G[dst][dst_nbr]["weight"]/p)
            elif G.has_edge(dst_nbr, src): #dtx=1
                unnormalized_probs.append(G[dst][dst_nbr]["weight"])
            else: #dtx = 2
                unnormalized_probs.append(G[dst][dst_nbr]["weight"]/q)
            
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
        
        return GEUtils.GEUtils.alias_setup(normalized_probs)
    
    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        
        G = self.G
        is_directed = self.is_directed
        
        alias_nodes = {}
        print("")
        print("Preprocessing.\n")
        for node in tqdm(G.nodes()):
            unnormalized_probs =  [G[node][nbr]["weight"] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = GEUtils.GEUtils.alias_setup(normalized_probs)
            
        alias_edges = {}
        traids = {}
        
        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in tqdm(G.edges()):
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge((edge[1], edge[0]))
                
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        
        return 
    
graphFile = "./graph/karate.edgelist"
karate_graph =  GEUtils.GEUtils.read_graph(graphFile, "edgelist")
GEUtils.GEUtils.draw_graph(karate_graph)
rand = RandomWalker(karate_graph,10,10)  
d, walks = rand.do_walk()
