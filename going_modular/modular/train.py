import torch
from torch import nn
from torchvision import transforms

import data_setup, model_builder, engine, utils

# Setup hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

train_loader, test_loader, class_names = data_setup.create_dataloaders(train_dir=train_dir, test_dir=test_dir, transform=data_transform, batch_size=BATCH_SIZE)
print(f'Data loaded, we got {len(train_loader.dataset)} train samples, and {len(test_loader.dataset)} test samples')
sample_image = next(iter(train_loader))[0][0]
(channel, height, weight) = sample_image.shape
print(f'Image size is {sample_image.shape}, {channel}, {height}, {weight}')

model = model_builder.MyVGG(num_in=3, 
                            num_out=len(class_names), 
                            num_hidden=HIDDEN_UNITS, 
                            image_height=height, 
                            image_weight=weight)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=LEARNING_RATE)

writer = utils.create_writer(experiment_name='my_exp', model_name='my_model')
engine.train(model=model, 
             train_loader=train_loader, 
             test_loader=test_loader, 
             epochs=NUM_EPOCHS, 
             loss_fn=loss_fn, 
             optimizer=optimizer,
             writer = writer)

utils.save_model(model, target_dir='models', model_name='MyVGG.pth')