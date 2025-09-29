import torch
# import torch.utils

print(torch.__version__)

print('cuda', torch.cuda.is_available())
print('mps', torch.backends.mps.is_available())
