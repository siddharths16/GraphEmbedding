# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:44:59 2019

@author: siddh
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from inputdata import Options, scorefunction
from sklearn.metrics.pairwise import euclidean_distances

class skipgram(nn.Module):
    
  def __init__(self, vocab_size, embedding_dim=16, cluster_number=20, gamma=0.5 ):
    super(skipgram, self).__init__()
    self.u_embeddings = nn.Embedding(vocab_size, embedding_dim)#, sparse=True)   
    self.v_embeddings = nn.Embedding(vocab_size, embedding_dim)#, sparse=True) 
    self.embedding_dim = embedding_dim
    self.vocab_size = vocab_size
    self.init_emb()
    self.gamma = gamma
    self.cluster_means = torch.FloatTensor(cluster_number, 
                                               embedding_dim).uniform_(-0.1/embedding_dim, 0.1/embedding_dim)
    
    #self.register_parameter('cluster_mean', nn.Parameter(self.cluster_means))
    self.cluster_means = self.cluster_means.cuda()
  def init_emb(self):
    initrange = 0.5 / self.embedding_dim
    self.u_embeddings.weight.data.uniform_(-initrange, initrange) ##u is input embedding
    self.v_embeddings.weight.data.uniform_(-0, 0)  ##context embedding
    
  def forward(self, u_pos, v_pos, v_neg, batch_size):

    embed_u = self.u_embeddings(u_pos) ##target matrix ?? 
    embed_v = self.v_embeddings(v_pos)
    
    score  = torch.mul(embed_u, embed_v)
    score = torch.sum(score, dim=1)  ##becomes: number of batches x window_size
    log_target = F.logsigmoid(score).squeeze()
    
    neg_embed_v = self.v_embeddings(v_neg) 
    #[batch_size, neg_size, emb_dim] x [batch_size, emb_dim, 1] = [batch_size, neg_size, 1]
    neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze(2)
    neg_score = torch.sum(neg_score, dim=1)
    sum_log_sampled = F.logsigmoid(-1*neg_score).squeeze()
    
    loss = log_target + sum_log_sampled
    
    clustering_distances =  torch.cdist(embed_u, self.cluster_means)
    clustering_loss = torch.sum(clustering_distances.min(dim=1)[0])
    clustering_loss = self.gamma*clustering_loss
    
    #print("skip-gram loss: ", loss.sum(), " clustering loss: ", clustering_loss)
    #print("Cluster means shape: ", self.cluster_means.shape)    
    ###Add clustering cost
    #print("Shape embed_u: ", embed_u.shape, " cluster_means: ", self.cluster_means.shape )
    #clustering_differences = embed_u.unsqueeze(1) - self.cluster_means
    #print("Clustering differences: ", clustering_differences.shape)
    
    #cluster_distances = clustering_differences.norm(p='fro', dim=2)
    #print("Cluster distance shape: ", cluster_distances.shape)
    
    #to_be_averaged = torch.min(cluster_distances, dim=1)[0]
    #print("to_be_averaged shape: ", to_be_averaged)
    
    #clustering_loss = self.gamma*torch.sum(to_be_averaged)  ###check if its sum or mean
    #print("skig-gram loss: ", loss.sum(), " Clustering cost: ", clustering_loss)
    ###Add Regularization cost
    
    #print("skip gram loss: ", loss.sum(), " clustering loss: ", clustering_loss ) #clustering loss is +ve and skip-gram is -ve
    loss_total = loss.sum()# + clustering_loss
    return -1*loss_total/batch_size
    
    #return -1*loss.sum()/batch_size

  def input_embeddings(self):
    return self.u_embeddings.weight.data.cpu().numpy()
    
  def get_cluster_means(self):
      return self.cluster_means.cpu().numpy()
    
  def save_embedding(self, file_name, id2word):
    embeds = self.u_embeddings.weight.data
    fo = open(file_name, 'w')
    for idx in range(len(embeds)):
      word = id2word(idx)
      embed = ' '.join(embeds[idx])
      fo.write(word+' '+embed+'\n')
