import torch
import torch.nn as nn 


class ConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding) -> None:
        super(ConvBlock, self).__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), # dimensional reduction
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True)
                             )
    def forward(self, x):
        return self.conv_layer(x)
        


class InceptionBlock(nn.Module):
    def __init__(self, in_ch,
                 num1x1, 
                 num_3x3_red, 
                 num_3x3,
                 num_5x5_red,
                 num_5x5,
                 pool_proj) -> None:
        super(InceptionBlock, self).__init__()

        '''
            4 branches 

        '''

        #branch 1: convolutional block with kernel size 1x1
        self.branch1 = ConvBlock(in_channels=in_ch, out_channels=num1x1, kernel_size=(1,1), stride=1, padding=0)

        #branch 2: 1x1 convolutional block followed by 3x3 conv block with stride 1
        self.branch2 = nn.Sequential(ConvBlock(in_channels=in_ch, out_channels=num_3x3_red, kernel_size=(1,1), stride=1, padding=0),
                                     ConvBlock(in_channels=num_3x3_red, out_channels=num_3x3, kernel_size=(3,3), stride=1, padding=1)
                                    )
        
        #branch 3: 1x1 conv block followed by 5x5 conv block with stride 2
        self.branch3 = nn.Sequential(ConvBlock(in_channels=in_ch, out_channels=num_5x5_red, kernel_size=(1,1), stride=1, padding=0),
                                     ConvBlock(in_channels=num_5x5_red, out_channels=num_5x5, kernel_size=(5,5), stride=1, padding=2)
                                    )
        
        #branch 4: Maxpool with 3x3 kernel with stride 1, followed by 1x1 conv block 
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=1, ceil_mode=True),
                                     ConvBlock(in_channels=in_ch, out_channels=pool_proj, kernel_size=(1,1), stride=1, padding=0)
                                    )


    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat((b1, b2, b3, b4), axis=1)

    

class GoogleNet(nn.Module):
    def __init__(self, inch_num, classes) -> None:
        super(GoogleNet, self).__init__()

        # initial layer initiate
        self.conv_layer1 = ConvBlock(in_channels=inch_num, out_channels=64, kernel_size=(7,7), stride=2, padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)
        self.conv_layer2 = nn.Sequential(ConvBlock(in_channels=64, out_channels=64, kernel_size=(1,1),stride=1, padding=0),
                                         ConvBlock(in_channels=64, out_channels=192, kernel_size=(3,3), stride=1, padding=1)
                                         )
        self.max_pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)
        # end of initial layers
        #start of Inception layers

        self.inception_layer1 = InceptionBlock(in_ch=192, num1x1=64, num_3x3_red=96, num_3x3=128, num_5x5_red=16, num_5x5=32, pool_proj=32)
        self.inception_layer2 = InceptionBlock(in_ch=256, num1x1=128, num_3x3_red=128, num_3x3=192, num_5x5_red=32, num_5x5=96, pool_proj=64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)
        
        self.inception_layer3 = InceptionBlock(in_ch=480, num1x1=192, num_3x3_red=96, num_3x3=208, num_5x5_red=16, num_5x5=48, pool_proj=64)
        self.inception_layer4 = InceptionBlock(in_ch=512, num1x1=160, num_3x3_red=112, num_3x3=224, num_5x5_red=24, num_5x5=64, pool_proj=64)
        self.inception_layer5 = InceptionBlock(in_ch=512, num1x1=128, num_3x3_red=128, num_3x3=256, num_5x5_red=24, num_5x5=64, pool_proj=64)
        self.inception_layer6 = InceptionBlock(in_ch=512, num1x1=112, num_3x3_red=144, num_3x3=288, num_5x5_red=32, num_5x5=64, pool_proj=64)
        self.inception_layer7 = InceptionBlock(in_ch=528, num1x1=256, num_3x3_red=160, num_3x3=320, num_5x5_red=32, num_5x5=128, pool_proj=128)
        self.max_pool4 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)

        self.inception_layer8 = InceptionBlock(in_ch=832, num1x1=256, num_3x3_red=160, num_3x3=320, num_5x5_red=32, num_5x5=128, pool_proj=128)
        self.inception_layer9 = InceptionBlock(in_ch=832, num1x1=384, num_3x3_red=192, num_3x3=384, num_5x5_red=48, num_5x5=128, pool_proj=128)



        # last layer
        self.final_layer = nn.Sequential(nn.AvgPool2d(kernel_size=(7,7), stride=1),
                                         nn.Dropout(0.4),
                                         nn.Flatten(),
                                        nn.Linear(in_features=1*1*1024, out_features=classes),
                                        nn.ReLU(inplace=True)
                                        )
    
    def forward(self, x):
        return self.__forward(x)
    
    def __forward(self, x):
        y = self.conv_layer1(x)
        y = self.max_pool1(y)
        y = self.conv_layer2(y)
        y = self.max_pool2(y)
        y = self.inception_layer1(y)
        y = self.inception_layer2(y)
        y = self.max_pool3(y)

        y = self.inception_layer3(y)
        y = self.inception_layer4(y)
        y = self.inception_layer5(y)
        y = self.inception_layer6(y)
        y = self.inception_layer7(y)
        y = self.max_pool4(y)

        y = self.inception_layer8(y)
        y = self.inception_layer9(y)

        y = self.final_layer(y)

        return y
    
    

if __name__ =="__main__":
    # test inception block
    # test input == (1,192, 28, 28)
    
    x = torch.rand((1, 192, 28, 28))
    ib_test  = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
    assert ib_test(x).shape == torch.Size([1, 256, 28, 28])
    print(ib_test(x).shape)

    # test model 
    x = torch.rand((1,3,224,224))
    model = GoogleNet(3, 10)
    print(model(x).shape)
    # print(model)