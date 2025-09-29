import vit
import torch
from torchinfo import summary

x = torch.randn(1, 3, 224, 224)

vit_layer = vit.ViT(num_classes=3)
vit_out = vit_layer(x)

print('vit_out = ', vit_out)

summary(vit_layer, 
        input_size=(32, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])
