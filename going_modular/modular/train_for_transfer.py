import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary

import data_setup, model_builder, engine, utils

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Create transforms
# data_transform = transforms.Compose([
#   transforms.Resize((64, 64)),
#   transforms.ToTensor()
# ])

# Option 1
# data_transform_transfer = transforms.Compose([
#   transforms.Resize((224, 224)),
#   transforms.ToTensor(),
#   transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
#                          std=[0.229, 0.224, 0.225])
# ])

# Option 2
data_transform_transfer = torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms()

train_loader, test_loader, class_names = data_setup.create_dataloaders(train_dir=train_dir, test_dir=test_dir, transform=data_transform_transfer, batch_size=BATCH_SIZE)
print(f'Data loaded, we got {len(train_loader.dataset)} train samples, and {len(test_loader.dataset)} test samples')
sample_image = next(iter(train_loader))[0][0]
(channel, height, weight) = sample_image.shape
print(f'Image size is {sample_image.shape}, {channel}, {height}, {weight}')

# model = model_builder.MyVGG(num_in=3, 
#                             num_out=len(class_names), 
#                             num_hidden=HIDDEN_UNITS, 
#                             image_height=height, 
#                             image_weight=weight)

# init a pretrained model
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)

# freeze the parameter
for param in model.features.parameters():
    param.requires_grad = False

# re-define the classifier feature
# print(f'old classifier, {model.classifier}')
# old classifier, Sequential(
#   (0): Dropout(p=0.2, inplace=True)
#   (1): Linear(in_features=1280, out_features=1000, bias=True)
# )
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
)

# summary of the model
# summary(model=model, 
#         input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
#         # col_names=["input_size"], # uncomment for smaller output
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=LEARNING_RATE)

writer = utils.create_writer(experiment_name='transfer_learning', model_name='efficientnet_b0', extra='test_close')

engine.train(model=model, 
             train_loader=train_loader, 
             test_loader=test_loader, 
             epochs=NUM_EPOCHS, 
             loss_fn=loss_fn, 
             optimizer=optimizer,
             writer=writer)

# utils.save_model(model, target_dir='models', model_name='MyVGG.pth')