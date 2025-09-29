from torch import nn
import msa_block, mlp_block

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 mlp_size:int=3072,
                 drop_out:float=0.1):
        super().__init__()
        self.msa_layer = msa_block.MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                  num_heads=num_heads)
        self.mlp_layer = mlp_block.MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, drop_out=drop_out)

    def forward(self, x):
        # residual connection
        x = self.msa_layer(x) + x
        x = self.mlp_layer(x) + x
        return x