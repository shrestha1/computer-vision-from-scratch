import torch
import torch.nn as nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self, in_ch, output) -> None:
        super(LeNet, self).__init__()
        self.model = nn.Sequential(

            nn.Conv2d(in_channels=in_ch, out_channels=6, kernel_size = 5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, out_features= output)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':

    # data: batchsize, channel, h, w
    x = torch.rand((1, 1, 32, 32))
    model = LeNet(1, 10)

    y = model(x)

    summary(model, (1,32,32))    
