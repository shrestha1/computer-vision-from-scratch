- paper: VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE SCALE IMAGE RECOGNITION
### Why VGG?
In case of Alex Net, it used large receptive window inorder to extract the feature from the given image. However, in VGG smaller receptive window of size 3x3 with stride 1 is used for feature extraction. The main reason to introduce VGG is to increase depth in the convolutional architecture and improve its performance.

VGG was introduced with different depth layers and it's varient are VGG11, VGG13, VGG16 and VGG19, where numbers are the convolutional layers.

VGG architecture performed with outstanding result in image recognition task.

Architecture
- Input RGB image of size 224x224, with only preprocessing involved is subtracting the mean RGB value, computed on the training set, from each pixel.
- Convolutional Layer: It depends on the form of VGG architecture to be designed.However the convolutional network is followed by ReLu and then maxpool of kernel size 2x2 along with stride of 2.
- Fully Connected Layer: This layer is same for every varient of VGG.  Two fully connected layer and a last layer which is capable of recognizing 1000 classes.
