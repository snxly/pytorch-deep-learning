import torch
from torch import nn
import patch_embedding, encoder

class ViT(nn.Module):
    def __init__(self,
                 image_size:int=224,
                 in_channel:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768,
                 embedding_dropout:float=0.1,
                 num_heads:int=12,
                 mlp_size:int=3072,
                 mlp_dropout:float=0.1,
                 num_tranformer_layer:int=12,
                 num_classes:int=1000,
                 ):
        super().__init__()

        assert image_size % patch_size == 0

        # Equation 1
        ## 1.1 patch embedding
        self.patch_embedding = patch_embedding.PatchEmbedding(in_channels=in_channel, 
                                                       patch_size=patch_size, 
                                                       embedding_dim=embedding_dim)
        ## 1.2 class embedding
        # 不同与之前的sample
        # 由于现在还不知道batch_size， 第一个dim先写成1， forward里再扩展
        class_data = torch.randn(1, 1, embedding_dim)
        self.class_embedding = nn.Parameter(class_data, requires_grad=True)

        ## 1.3 position embedding
        # 注意这里 '//' 的用法，代替了 int()
        self.num_patch = image_size * image_size // (patch_size * patch_size)
        position_data = torch.randn(1, self.num_patch+1, embedding_dim)
        self.position_embedding = nn.Parameter(position_data, requires_grad=True)

        ## 1.4 dropout for embedding layer
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Equation 2 and 3
        encoder_layers = [
            encoder.TransformerEncoderBlock(embedding_dim=embedding_dim, 
                                            num_heads=num_heads,
                                            mlp_size=mlp_size,
                                            drop_out=mlp_dropout)
            for i in range(num_tranformer_layer)
        ]
        self.encoder_layer = nn.Sequential(*encoder_layers)

        # Equation 4 MLP Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # expand 有点像广播前的准备工作，提前拷贝
        # 实际上并没有做真正的拷贝，只是把dim改了
        # Q: 不做这个expand是不是也可以自动广播？
        class_embedding = self.class_embedding.expand(batch_size, -1, -1)

        x = self.patch_embedding(x)
        x = torch.cat((class_embedding, x), dim=1)
        x = x + self.position_embedding

        x = self.embedding_dropout(x)

        x = self.encoder_layer(x)

        # Q: x 有三个维度，这里怎么只写了两个？
        # A: 等同于 x[:,0,:]
        x = self.classifier(x[:, 0])

        return x

















