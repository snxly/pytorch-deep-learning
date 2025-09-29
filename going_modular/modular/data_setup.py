import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

num_workers = os.cpu_count()
# print(f'cpu count is {num_workers}')

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_data, 
                              batch_size=batch_size,
                              shuffle=True, 
                        #       num_workers=num_workers,
                              pin_memory=True,
                              )
    
    test_loader = DataLoader(test_data, 
                              batch_size=batch_size,
                              shuffle=True, 
                        #       num_workers=num_workers,
                              pin_memory=True,
                              )
    
    class_names = train_data.classes

    return train_loader, test_loader, class_names