## Introduction 
LeNet-5 is implementation of convolutional neural network used primarily to recognize characters and this network was introduced and implemented by LeCunn. 

Architecture
LeNet comprises 7 layers, not counting the input, all of which contains trainable parameters(weights). 

Input: 32x32 pixel image

pixel values are normalized so that the background level(white) corresponds to a value of -0.1 and foreground (black) corresponds to 1.175.

Layer 1 C1: 
-  Convolutional layer with 6 feature maps of size 28x28.
-  5x5 filter is used.
<!-- -  trainable parameters 156 and 122,304 connections. -->
-  Hyperbolic tangent is used as an activation function

Layer 2 S2:
-  Generates 6 feature maps of size 14x14.
-  2x2 filter is used.
-  Stride used 2 for non overlapping
<!-- -  12 trainable parametersand 5,880 connections -->

Layer 3 C3:
-  Convolutional layer generates 16 feature maps.
-  5x5 filter is used.
<!-- -  trainable parameters 1516 and 1516,000 connections -->
-  Hyperbolic tangent is used as an activation function.

Layer 4 S4:
-  Generates 16 feature maps of size 5x5.
-  2x2 filter is used.
<!-- -  trainable parameters 32 and 2000 connections -->

Layer 5 C5:
-  Convolution layer generates 120 feature maps of size 1x1.
-  5x5 filter is used.
<!-- -  48120 trainable connections. -->

Layer 6 F6:
-  input 120 and output 84
<!-- -  trainable parameters 10164 -->
-  ReLU as an activation function

Layer 7 F7:
- output layer is originally a RBF in paper LeNet-5, however a Linear layer is used in this model



## Reference 
- http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf