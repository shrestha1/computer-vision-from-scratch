import torch
import torch.nn as nn

############################
#       AlexNet
############################

class AlexNet(nn.Module):
    def __init__(self, n_ch, out_ch, drop_out) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=n_ch, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=256, out_channels=348, kernel_size=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=348, out_channels=348, kernel_size=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=348, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=drop_out),
            nn.Linear(256*5*5, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=drop_out),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, out_ch)
        )

    def forward(self, x):
        f = self.features(x)      #Extract the features
        f = torch.flatten(f, 1)   #Flatten the tensor
        return self.fc_layer(f)   #return from the last fully connected layer


## TODO: Network not compatible with dataset. The network have to be maintained
class TIAlexNet(nn.Module):
    def __init__(self, in_ch, out_ch, drop_out) -> None:
        super(TIAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels = 96, kernel_size=5, stride=3 , padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            
            nn.Conv2d(in_channels=256, out_channels=348, kernel_size=3),
            nn.ReLU(inplace=True),
                       
            nn.Conv2d(in_channels=348, out_channels=348, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
                        
            nn.Conv2d(in_channels=348, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=drop_out),
            nn.Linear(256*10*10, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=drop_out),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, out_ch)
        )

    def forward(self, x):
        f = self.features(x)      #Extract the features
        f = torch.flatten(f, 1)   #Flatten the tensor
        return self.fc_layer(f)   #return from the last fully connected layer

if __name__=='__main__':
    #data: batch, channel, h, w
    x = torch.randn(1, 3, 227, 227)
    model = AlexNet(3, 10, 0.5)
    y = model(x)

    x = torch.randn(1, 3, 64, 64)
    model = TIAlexNet(3, 10, 0.5)
    y = model(x)
    print(y)