import os
import sys
import numpy as np
import torch
import torch.nn as nn
import math
sys.path.append(os.path.join(os.path.dirname(__file__), "./.."))
from src.Layer import Layer
#from Layer import Layer

def get_activation(activation, slope=0.1):
   if activation == "relu":
       return nn.ReLU()
   elif activation == "elu":
       return nn.ELU()
   elif activation == "lrelu":
       return nn.LeakyReLU(slope)
   elif activation == "swish":
       return nn.SiLU()
   else:
       raise KeyError(activation)


class Block(nn.Module):
  def __init__(self, 
               embed_dim,
               d_model = 32,
               nhead = 8,
               dim_feedforward = 128,
               dropout = 0.1,
               layer_norm_eps = 1e-5,
               activation = "lrelu",
               add_1D_embed = True):
       
       super(Block, self).__init__()
               
       self.embed_1d = add_1D_embed
       
       # Transformer layer.
       self.attn = Layer(in_dim=embed_dim, d_model=d_model, nhead=nhead)
                  
       # Updater module (implementation of Feedforward model of the original transformer)
       if self.embed_1d:
           updater_in_dim = embed_dim + d_model
       else:
           updater_in_dim = d_model
           
           
       self.linear1 = nn.Linear(updater_in_dim, dim_feedforward)
       self.linear2 = nn.Linear(dim_feedforward, embed_dim)
       self.dropout = nn.Dropout(dropout)
       self.activation = get_activation(activation)
       
       # FIXED: Create Sequential after defining the layers
       self.update_module = nn.Sequential(
           self.linear1, 
           self.activation,
           self.dropout, 
           self.linear2
       )

       self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
       self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

 
  def forward(self, s, x, rel_p):
       """
       
       """
       
       # att, skip, norm
       s1 = self.attn(s, rel_p)                   # self attention
       s = s + self.dropout(s1)            # dropout and add ( skip con type)   
       s = self.norm1(s)                  # use normalization


       # concat 1D and output along last dim
       if self.embed_1d:
           um_in = torch.cat([s, x], dim=-1) 
       else:
           um_in = s
           

       #update module    
       s2 = self.update_module(um_in)
               
       #  skip, norm
       s = s + s2 #self.dropout(s2)         # dropout and add ( skip con type) 
       s = self.norm2(s)                # norm

       return s

   
if __name__ == "__main__":
   
   seq_len = 10         # seq length
   nhead = 8            # numper of attention heads
   d_model = 32         # embedding dimention, must be = c*n_head ( c = head dim) 

   in_dim_2d = d_model  # inp dim 
   batch_size = 2  
   in_dim = d_model 
   nhead = 8
   np = d_model
   
   # instantiate model        
   GANB = Block(embed_dim = d_model, nhead = nhead, add_1D_embed = True) 
  
   # prep dat
   inp = torch.rand(seq_len, batch_size, d_model)
   x = torch.rand(seq_len, batch_size, np)           # 1d sequence embedding  

   rel_p = torch.rand(seq_len, batch_size, seq_len, np)

   # # forward pass
   out = GANB(inp, x, rel_p)
   
   print(out.shape)