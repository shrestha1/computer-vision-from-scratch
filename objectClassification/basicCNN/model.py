'''
Basic: Three CNN layer followed by a hidden linear layer along with output linear layer

'''

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_ch, classes) -> None:
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=num_ch, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, classes)
        )

    def forward(self, x):
        return self.model(x)
    

if __name__ == '__main__':
    # batch size, channel, h, w 
    x = torch.rand((1, 1, 32, 32))
   
    model = CNN(1, 10)
    y = model(x)
    print(y)
    from torchsummary import summary
    summary(model, (1,32,32))


    