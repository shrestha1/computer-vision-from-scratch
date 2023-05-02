import torch 
import torch.nn as nn

class BuildingBlock(nn.Module):
    '''
    As per paper building block in 2 layer architecture
    with its input added with the out of last layer
    - 3x3 convolution

    '''
    def __init__(self, in_ch, out_ch, stride) :
        super(BuildingBlock, self).__init__()
        self.stride = stride
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size= 3, stride=stride, padding=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size= 3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_ch)
                                   )
        
    def forward(self, x):
        out = self.block(x)
        if x.shape != out.shape:
            x = nn.Conv2d(x.shape[1], out.shape[1], kernel_size=1, stride=self.stride)(x)
            x = nn.BatchNorm2d(x.shape[1])(x)

        out+= x

        return torch.relu(out)


class Resnet(nn.Module):
    def __init__(self, config, classes=1000):
        super().__init__()
        self.config = config
        self.in_ch = 64
        self.classes = classes

        self.layer0 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                                    nn.BatchNorm2d(64)
                                    )
        self.maxpool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__make_layer( 64, 1, self.config[0])
        self.layer2 = self.__make_layer( 128, 2, self.config[1])
        self.layer3 = self.__make_layer( 256, 2, self.config[2])
        self.layer4 = self.__make_layer( 512, 2, self.config[3])
        self.average_pool = nn.AvgPool2d((1,1))
        
        self.fc_layer = nn.Linear(512*7*7, self.classes)
        
    def forward(self, x):
        out = self.layer0(x)

        out = self.maxpool0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.average_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc_layer(out) 
       
        return out

    def __make_layer(self, out_ch, starting_stride, nth_layer:int):
        layer = []
        strides = [starting_stride] + [1]*(nth_layer-1)

        for stride in strides:
            layer.append(BuildingBlock(self.in_ch, out_ch, stride))
            self.in_ch = out_ch
        
        return nn.Sequential(*layer)
    
    @staticmethod
    def resnet34():
        config = [3, 4, 6, 3]
        return Resnet(config=config)
    

if __name__ == '__main__':
    x = torch.rand((1, 3, 224, 224))
    y = Resnet.resnet34()

    print(y)
    # print(y)