import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, config, out_ch=1000):
        super(VGG, self).__init__()    

        self.features = self.create_conv_layer(config=config)
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
    
    def create_conv_layer(self, config):
        in_channel = 3
        layers = []

        for c in config:
            if c == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                                nn.Conv2d(in_channels=in_channel,
                                                     out_channels= c, 
                                                     kernel_size=3, 
                                                     stride=1, 
                                                     padding=1),
                                nn.ReLU(inplace=True)])
                in_channel = c
        return nn.Sequential(*layers)
        
    @staticmethod
    def vgg11():
        config = [64,'M', 128, 'M', 256, 256,'M', 512, 512,'M', 512, 512,'M']
        return VGG(config)
    
    @staticmethod
    def vgg13():
        config = [64, 64,'M', 128, 128, 'M', 256, 256,'M', 512, 512,'M', 512, 512,'M']
        return VGG(config)
    
    @staticmethod
    def vgg16():
        config = [64, 64,'M', 128, 128, 'M', 256, 256, 256,'M', 512, 512, 512,'M', 512, 512, 512,'M']
        return VGG(config)
    
    @staticmethod
    def vgg19():
        config = [64, 64,'M', 128, 128, 'M', 256, 256, 256, 256,'M', 512, 512, 512, 512,'M', 512, 512, 512, 512,'M']
        return VGG(config)
    
if __name__ == '__main__':
    # x = batch, channel , height, width
    x = torch.rand((1, 3, 224, 224))

    model = VGG.vgg16()

    # print(model(x))
    from torchsummary import summary
    summary(model, (3, 224, 224))
    
    from torchvision import models
    vgg = models.vgg16()
    summary(vgg, (3, 224, 224))