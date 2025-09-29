import torch
from torch import nn
import torchvision
# print(f'torch version = {torch.__version__}, torchvision version = {torchvision.__version__}')

import sys
import os

# 获取项目根目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir = {current_dir}')
project_root = os.path.dirname(current_dir)
# print(f'project_root = {project_root}')
sys.path.append(project_root)
from going_modular.modular import helper_functions, data_setup, engine, utils
# helper_functions.import_test()

import models
from torchinfo import summary
from pathlib import Path

# Download pizza, steak, sushi images from GitHub
data_20_percent_path = helper_functions.download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                     destination="pizza_steak_sushi_20_percent")

# print(data_20_percent_path)

# Setup directory paths to train and test images
train_dir = data_20_percent_path / "train"
test_dir = data_20_percent_path / "test"

# setup model 1: Efficient_b2
efficient_b2_model, efficient_b2_transform  = models.create_efficient_b2()

# summary(efficient_b2_model, 
#         input_size=(1, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

batch_size=32 
train_loader_effb2, test_loader_effb2, class_names_effb2 = data_setup.create_dataloaders(train_dir=train_dir, 
                                                                test_dir=test_dir, 
                                                                transform=efficient_b2_transform,
                                                                batch_size=batch_size)

print('class_names=', class_names_effb2)
pass

# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(params=efficient_b2_model.parameters(),
#                              lr=1e-3)

# efficent_b2_result = engine.train(model=efficient_b2_model,
#                                 train_loader=train_loader_effb2,
#                                 test_loader=test_loader_effb2,
#                                 loss_fn=loss_fn,
#                                 optimizer=optimizer,
#                                 epochs=10)

# # Save the model
# utils.save_model(model=efficient_b2_model,
#                  target_dir="models",
#                  model_name="efficient_b2.pth")

# model_path = 'models/efficient_b2.pth'

# effnetb2_model_size = Path(model_path).stat().st_size // (1024*1024) # division converts bytes to megabytes (roughly) 
# print(f"Pretrained EffNetB2 feature extractor model size: {effnetb2_model_size} MB")

# evaluate
model, model_transform = models.load_efficient_b2()

train_loader_effb2, test_loader_effb2, class_names_effb2 = data_setup.create_dataloaders(train_dir=train_dir, 
                                                                test_dir=test_dir, 
                                                                transform=model_transform,
                                                                batch_size=batch_size)

with torch.inference_mode():
    X, y = next(iter(test_loader_effb2))

    logits = model(X)
    prob = torch.softmax(logits, dim=1)
    pred = prob.argmax(dim=1)

    print(f'acc = {(pred == y).sum().item()} / {batch_size}')
    print(prob.max(dim=1))
        

