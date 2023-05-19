import torch
import torch.nn as nn 


class Inception(nn.Module):
    def __init__(self, in_ch) -> None:
        super(Inception, self).__init__()

        '''
            4 branches 

        '''

        #branch 1: convolutional block with kernel size 1x1
        self.branch1 = nn.Conv2d(in_channels=in_ch, out_channels=..., kernel_size=(1,1), stride=1)

        #branch 2: 1x1 convolutional block followed by 3x3 conv block with stride 1
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=..., kernel_size=(3,3), stride=1),
                                     nn.Conv2d(in_channels=..., out_channels=..., kernel_size=(3,3), stride=1)
                                    )
        
        #branch 3: 1x1 conv block followed by 5x5 conv block with stride 2
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=..., kernel_size=(1,1)),
                                     nn.Conv2d(in_channel=..., out_channels=..., kernel_size=(5,5), stride=2)
                                    )
        
        #branch 4: Maxpool with 3x3 kernel with stride 1, followed by 1x1 conv block 
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=(3,3), stride=1),
                                    nn.Conv2d(in_channels=..., out_channels=..., kernel_size=(1,1))
                                    )


    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat((b1, b2, b3, b4), axis=1)

    
    

class GoogleNet(nn.Module):
    def __init__(self) -> None:
        super(GoogleNet, self).__init__()

        self.initial_layer = nn.Sequential(nn.Conv2d(in_channels=..., out_channels=..., kernel_size=(7,7), stride=2),
                                          nn.MaxPool2d(kernel_size=(3,3), stride=2),
                                          nn.LocalResponseNorm(2),
                                          nn.Conv2d(in_channels=..., out_channels=..., kernel_size=(1,1), stride=1),
                                          nn.Conv2d(in_channels=..., out_channels=..., kernel_size=(3,3), stride=1),
                                          nn.LocalResponseNorm(2),
                                          nn.MaxPool2d(kernel_size=(3,3), stride=2)
                                          )
        
        # single inception block
        self.inception = Inception(in_ch=...)

        # last layer
        self.final_layer = nn.Sequential(nn.AvgPool2d(kernel_size=(7,7), stride=1),
                                        nn.Linear(in_features=..., out_features=...)
                                        
                                        )