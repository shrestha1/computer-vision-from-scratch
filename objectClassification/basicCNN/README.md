## BACKGROUND
Images are made of pixels and are  unstructured datasets since the pixels intensity varies from one to another image of same object. There might not be any order in the pixels values, which are the features to be acknowledged while performing object classification. Nearby pixels are highly correlated and thus these local features are to be extracted and combined before recognizing the structures of the objects(ex: edges, corners, etc). These features extraction can be done automatically using convolution operation with appropriate filters.

## INTRODUCTION
Convolution Operation is implemented to extract features from image applying filters; some n x n matrix which has a significant values to represent its function. Convolutional Neural Network is the network in which its layers have convolution operation implemented.

CNN combines three architectural ideas to ensure some degree of shift, scale and distortion invariance: local receptive field, shared weights and spatial or temporal sub-sampling.
