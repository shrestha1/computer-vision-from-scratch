import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, n_ch, out_ch, drop_out) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=n_ch, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),


        )


if __name__=='__main__':
    #data: batch, channel, h, w
    x = torch.randn(1, 3, 227, 227)
    model = AlexNet(3, 10, 0.5)
    y = model(x)
    