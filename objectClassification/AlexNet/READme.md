- Paper: ImageNet Classification with Deep Convolutional Neural Networks

## AlexNet:

- 5 convolutional layers, some of which are followed by max pooling layers and three fully connected layers with 1000 way softmax
- To make the training Faster (originally in paper):
    - Non saturating neurons are used
    - GPU is used
- To reduce overfitting in the FC layers: 
    - Regularization technique called dropout is used

- The capacity of convolutional network can be varied by varying its depth and breath
- ReLU is introduced as non-saturating nonlinearity instead of saturating nonlinearity tanh
- Parallelization scheme was introduced. 

## Local Response Normalization
If any neuron has positive input to ReLU then that particular neuron will be cable of learning.
It is a technique used in convolution neural networks to aid in generalization. It involves applying a normalization scheme to the output of a neuron, which creates competition for big activities amongst neuron outputs computed using different kernels. The constant used in the normalization scheme is determined by using a validation set. 

## Overlapping Pooling
Overlapping pooling is used to summarize the outputs of neighboring neurons in the same kernel maps
Models with overlapping poolings are slightly more difficult to overfit during training

## Architecture of AlexNet (originally)

- **Input image** size: 227 x 227 x 3
- **First layer**     : 96 kernels of size 11 x 11 x 3 with stride of 4 
- **Second layer**    : 256 kernels of size 5x5x48
**3, 4 and 5th conv layers: no pooling**
- **3 layer**         : 384 kernels of size 3 X 3 X 256
- **4th layer**       : 384 kernels of size 3x3x192
- **5th layer**       : 256 kernels of size 3x3x192 
- **FC1 layer**       : 256*5*5 input to 4096 out
- **FC2 layer**       : 4096 input to 4096 out
- **FC3 layer**       : 4096 input to 10 out 


### Modified AlexNet for  the TinyImageNet Datasets
TinyImageNet datasets has training images collection for **200** classes and the size of image is **64x64**. Thus it is required to modify the original architecture of AlexNet to train the TinyImageNet dataset.