from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channels:int=3,
                 embedding_dim:int=768,
                 patch_size:int=16):
        super().__init__()

        self.patch_size = patch_size
        
        self.conv2d = nn.Conv2d(in_channels=in_channels, 
                   kernel_size=patch_size,
                   stride=patch_size,
                   out_channels=embedding_dim)
        
        self.flattern = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        # [1, 768, 14, 14]
        x_conv = self.conv2d(x)
        # [1, 768, 196]
        x_flattern = self.flattern(x_conv)

        # [1, 196, 768]
        return x_flattern.permute(0, 2, 1)
