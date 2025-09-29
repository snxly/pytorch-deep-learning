import torchvision
import torch
from torch import nn

def create_efficient_b2():
    b2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    efficient_b2 = torchvision.models.efficientnet_b2(weights=b2_weights)
    b2_tranform = b2_weights.transforms()

    for param in efficient_b2.parameters():
        param.requires_grad = False
    # print(efficient_b2.classifier)
    efficient_b2.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        # change out_features, 1000 -> 3
        nn.Linear(in_features=1408, out_features=3, bias=True),
    )

    return efficient_b2, b2_tranform

def load_efficient_b2(model_path:str='models/efficient_b2.pth'):
    # load pretrained model
    model_weights = torch.load(model_path)

    model, model_transform = create_efficient_b2()
    model.load_state_dict(model_weights)

    return model, model_transform
