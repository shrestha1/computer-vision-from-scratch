import torch
import torch.nn as nn


class VGG11(nn.Module):
    def __init__(self, n_ch, out_ch):
        
        __n_layers = [64,'M', 128, 'M', 256, 256,'M', 512, 512,'M', 512, 512,'M']
        
        super(VGG11, self).__init__()
        self.feature_list = []
        
        self.n_ch = n_ch
        for ch in __n_layers:
            if ch == 'M':
                self.feature_list.append(nn.MaxPool2d(2, 2))
            else:
                self.feature_list.extend([nn.Conv2d(in_channels=self.n_ch,
                                                     out_channels= ch, 
                                                     kernel_size=3, 
                                                     stride=1, 
                                                     padding=1),
                                          nn.ReLU(),
                                    ])
                self.n_ch = ch
            
        self.features = nn.Sequential(*self.feature_list)

        self.fc_layer = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(4096, out_ch)
        )

    def forward(self, x):
        f = self.features(x)
        ## flatten the f to match with fc layer
        f = torch.flatten(f, 1) 
        return self.fc_layer(f)


if __name__ == '__main__':
    # x = batch, channel , height, width
    x = torch.rand((1, 3, 224, 224))

    model = VGG11(3, 1000)

    # print(model(x))
    from torchsummary import summary
    summary(model, (3, 224, 224))
    
    from torchvision import models
    vgg = models.vgg11()
    summary(vgg, (3, 224, 224))