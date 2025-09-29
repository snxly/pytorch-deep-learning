from torch import nn

class MyVGG(nn.Module):
    def __init__(self, num_in, num_out, num_hidden, image_height, image_weight):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=num_in,
                      out_channels=num_hidden,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_hidden,
                      out_channels=num_hidden,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden,
                      out_channels=num_hidden,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_hidden,
                      out_channels=num_hidden,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # @todo: can we change this hardcode?
        num_flattern = int(num_hidden * (image_height / 4) * (image_weight / 4))
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_flattern, out_features=num_out)
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classify(x)
        return x