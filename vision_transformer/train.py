import sys
import os

# 获取项目根目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir = {current_dir}')
project_root = os.path.dirname(current_dir)
# print(f'project_root = {project_root}')
sys.path.append(project_root)

from going_modular.modular.helper_functions import download_data
from going_modular.modular import data_setup, engine
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torch import nn
import patch_embedding, msa_block, mlp_block, encoder
from torchinfo import summary
import vit

# download data
image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
# print(image_path)
train_path = image_path / 'train'
test_path = image_path / 'test'

IMAGE_SIZE = 224
# setup transform
manual_tranform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),                  
])

# create dataloaders
BATCH_SIZE = 32
train_loader, test_loader, class_names = data_setup.create_dataloaders(train_dir=train_path, 
                              test_dir=test_path, 
                              transform=manual_tranform, 
                              batch_size=BATCH_SIZE)
print(f'{len(train_loader.dataset), len(test_loader.dataset), class_names}')

# create model
model = vit.ViT(num_classes=len(class_names))

# # Visualize image
# image_batch, label_batch = next(iter(train_loader))
# sample_image = image_batch[0]
# sample_label = label_batch[0]
# print(sample_image.shape, sample_label)

# # CHW -> HWC
# plt_image = sample_image.permute(1,2,0)

# # plt.imshow(plt_image)
# # plt.title(class_names[sample_label])
# # plt.axis(False)
# # plt.show()

# # Equation 1, input and outpyt shape of patch-embedding layer
# # given path_size, get patch number
# height = 224
# weight = 224
# patch_size = 16 # small image is 16 * 16
# color_channels = 3

# number_of_patches = int(height * weight / (patch_size * patch_size))
# print(f'One image will be sliced into {number_of_patches} patches, with size {patch_size} * {patch_size}')

# # input and output shape of embedding
# # (224, 224, 3) --> (number_of_patches, ??)
# embedding_layer_input_shape = (height, weight, color_channels)
# embedding_layer_output_shape = (number_of_patches, patch_size ** 2 * color_channels)
# print(f'embedding_layer_input_shape is {embedding_layer_input_shape}, embedding_layer_output_shape is {embedding_layer_output_shape}')

# # Turn sample_image into patches
# # num_patches = int(height / patch_size)
# # fig, axs = plt.subplots(nrows=num_patches, ncols=num_patches,
# #                        figsize=(num_patches * 0.5, num_patches * 0.5),
# #                         sharex=True,
# #                         sharey=True)

# # for row, patch_row in enumerate(range(0, IMAGE_SIZE, patch_size)):
# #     for col, patch_col in enumerate(range(0, IMAGE_SIZE, patch_size)):
# #         ax = axs[row][col]
# #         ax.imshow(plt_image[patch_row:patch_row+patch_size, patch_col:patch_col+patch_size, :]); # keep height index constant, alter the width index
# #         # ax.set_xlabel(i+1) # set the label
# #         ax.set_xticks([])
# #         ax.set_yticks([])

# # plt.show()

# # Create the model

# embedding_dim = embedding_layer_output_shape[1]
# # Patched Embedding
# patch_embedding_layer = patch_embedding.PatchEmbedding(in_channels=3, 
#                                                        patch_size=patch_size, 
#                                                        embedding_dim=embedding_dim)

# patch_embedding_out = patch_embedding_layer(sample_image.unsqueeze(0))
# # [1, 196, 768]
# print('patch_embedding_out.shape = ', patch_embedding_out.shape)

# # summary(patch_embedding_layer, 
# #         input_size=(1, color_channels, height, weight))

# # class embedding
# # [1, 196, 768] => [1, 197, 768]
# (class_batch, class_patch_number, class_embedding) = patch_embedding_out.shape
# class_data = torch.randn(class_batch, 1, class_embedding)
# class_param = nn.Parameter(class_data, requires_grad=True)

# patch_embedded_image_with_class_embedding = torch.cat((class_param, patch_embedding_out),
#                                                       dim=1)
# # [1, 197, 768]
# print('patch_embedded_image_with_class_embedding.shape = ', patch_embedded_image_with_class_embedding.shape)

# # position embedding
# (pos_batch, pos_patch_number, pos_embedding) = patch_embedded_image_with_class_embedding.shape
# pos_data = torch.randn(1, pos_patch_number, pos_embedding)
# position_embedding = nn.Parameter(pos_data, requires_grad=True)

# patch_and_position_embedding = patch_embedded_image_with_class_embedding + position_embedding
# # print(patch_and_position_embedding)
# print(f"Patch embeddings, class token prepended and positional embeddings added shape: {patch_and_position_embedding.shape} -> [batch_size, number_of_patches, embedding_dimension]")

# num_heads = 12
# # msa block
# # msa_layer = msa_block.MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
# #                                                   num_heads=num_heads)
# # msa_out = msa_layer(patch_and_position_embedding)
# # print('msa_out.shape=', msa_out.shape)

# # mlp block
# mlp_size = 3072
# drop_out = 0.1
# # mlp_layer = mlp_block.MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, drop_out=drop_out)
# # mlp_out = mlp_layer(msa_out)
# # print('mlp_out.shape = ', mlp_out.shape)

# encoder_layer = encoder.TransformerEncoderBlock(embedding_dim=embedding_dim, 
#                                                 num_heads=12, 
#                                                 mlp_size=mlp_size, 
#                                                 drop_out=drop_out)
# encoder_out = encoder_layer(patch_and_position_embedding)
# print('encoder_out.shape = ', encoder_out.shape)

# summary(model=encoder_layer, 
#         input_size=(1, 197, 768))

# Train the model
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=3e-3, # Base LR from Table 3 for ViT-* ImageNet-1k
                             betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                             weight_decay=0.3)

results = engine.train(model=model,
                       train_loader=train_loader,
                       test_loader=test_loader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=10)

# Evaluate the model

# Inference on the model