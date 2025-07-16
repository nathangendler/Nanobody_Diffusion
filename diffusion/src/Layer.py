import numpy as np
import torch
import torch.nn as nn
import math

class Layer(nn.Module):
    
    def __init__(self, in_dim, d_model, nhead):
        super(Layer, self).__init__()
        """d_model = c*n_head"""
        
        head_dim = d_model // nhead
        assert head_dim * nhead == d_model, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = head_dim
        
        
        # Linear layers for q, k, v for dot product affinities.
        self.q_linear = nn.Linear(in_dim, self.d_model, bias=False)
        self.k_linear = nn.Linear(in_dim, self.d_model, bias=False)
        self.v_linear = nn.Linear(in_dim, self.d_model, bias=False)
        
        
        # 2d representation.
        self.mlp_2d = nn.Linear(self.d_model, self.nhead, bias=False)
        
        # Output layer.
        self.out_linear = nn.Linear(self.d_model, in_dim)
        
      
    def forward(self, s, rel_p):
        """
        Receives s (L, N, E) tensor.
            L: sequence length,
            N: batch size,
            E: input embedding dimension.
               
        """
        
        seq_l, b_size, _ = s.shape
        
        #----------------------------------------------
        # Compute q, k, v vectors. Will reshape to (L, N, D*H).
        # D: number of dimensions per head,
        # H: number of head,
        # E = D*H: embedding dimension.
        q = self.q_linear(s)
        k = self.k_linear(s)
        v = self.v_linear(s)

        # Actually compute dot prodcut affinities.
        # Reshape first to (N*H, L, D).
        q = q.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)

        # Then perform matrix multiplication between two batches of matrices.
        # (N*H, L, D) x (N*H, D, L) -> (N*H, L, L)
        score = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.d_model) 
        #print('score -- ', score.shape)
        
        
        # Use the 2d branch (rel_p)
        rel_p = self.mlp_2d(rel_p)
        # (N, L1, L2, H) -> (N, H, L2, L1)
        rel_p = rel_p.transpose(1, 3)
        # (N, H, L2, L1) -> (N, H, L1, L2)
        rel_p = rel_p.transpose(2, 3)
        # (N, H, L1, L2) -> (N*H, L1, L2)
        rel_p = rel_p.contiguous().view(b_size*self.nhead, seq_l, seq_l)
        
        
        #--------------------------------
        # Compute the attention values. -
        #--------------------------------
        #softmax
        attn = nn.functional.softmax(score + rel_p, dim=-1)
        
        # Update values obtained in the dot product affinity branch.
        s_new = torch.bmm(attn, v)
        
        # Reshape the output, that has a shape of (N*H, L, D) back to (L, N, D*H).
        s_new = s_new.transpose(0, 1).contiguous().view(seq_l, b_size, self.d_model)
        
        return self.out_linear(s_new)

    
if __name__ == "__main__":
    
    seq_len = 10           # seq length
    nhead = 8              # numper of attention heads
    d_model = 32           # embedding dimention, must be = c*n_head ( c = head dim) 

    in_dim_2d = d_model  # inp dim 
    batch_size = 2  
    in_dim = d_model 
    nhead = 8
    np = d_model
    
    # instantiate model        
    model = Layer(in_dim, d_model, nhead)
    
    # prep dat
    inp = torch.rand(seq_len, batch_size, d_model)
    
    rel_p = torch.rand(seq_len, batch_size, seq_len, np)
    
    # forward pass
    out = model(inp, rel_p)
    
    print(out.shape)

        
