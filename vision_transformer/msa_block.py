import torch
from torch import nn

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,          # hyperparameter
                 att_dropout:float=0):      # hyperparameter
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.msa = nn.MultiheadAttention(embed_dim=embedding_dim, 
                                         num_heads=num_heads,
                                         dropout=att_dropout,
                                         batch_first=True) # batch is the fist dimention
    
    def forward(self, x):
        x = self.layer_norm(x)

        output, output_weights = self.msa(query=x, # q
                                            key=x,   # k
                                            value=x, # v
                                            need_weights=False) # do we need weights, or just layer output?
        
        # todo: residual connections later
        return output