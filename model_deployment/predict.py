import torch
import models
import torchvision
import sys
import os
from pathlib import Path
import random

# 获取项目根目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir = {current_dir}')
project_root = os.path.dirname(current_dir)
# print(f'project_root = {project_root}')
sys.path.append(project_root)
from going_modular.modular import helper_functions, data_setup, engine, utils

model, model_transform = models.load_efficient_b2()
class_names= ['pizza', 'steak', 'sushi']

def predict(img, model=model, model_transform=model_transform):
    img = model_transform(img).unsqueeze(0)
    with torch.inference_mode():
        logits = model(img)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)

        pred = pred.item()
        prob = prob.max(dim=1).values.item()

    return class_names[pred], prob

def get_sample_image(num:int=1):
    # Get a list of all test image filepaths
    test_data_paths = list(Path('data/pizza_steak_sushi_20_percent/test').glob("*/*.jpg"))

    # Randomly select a test image path
    result = [[str(file_path)] for file_path in random.sample(test_data_paths, k=num)]

    return result

print(get_sample_image(1))


