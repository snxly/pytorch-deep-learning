from torch import nn

patch_size = 16

conv2d = nn.Conv2d(in_channels=3, 
                   kernel_size=patch_size,
                   stride=patch_size,
                   # todo: we've calculate this
                   out_channels=768)