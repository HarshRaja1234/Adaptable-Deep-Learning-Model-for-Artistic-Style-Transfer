# Adaptable-Deep-Learning-Model-for-Artistic-Style-Transfer
this assignment is to create a deep learning model capable of adapting an existing work to resemble the aesthetic of any art. 
## Overview
Neural style transfer is an intriguing technique that combines the content of one image with the artistic style of another, resulting in a unique and visually appealing synthesis. This process adds an artistic touch to images, diverging from conventional photography to embrace non-photorealistic rendering.

### Working Methodology
#### Network Utilization:
Employing a pre-trained VGG19 convolutional neural network, we capture intricate image details. Early layers focus on 'style' elements like colors, while later layers depict more complex 'content' details such as specific object features.

#### Activation Extraction:
Inputting the content image into VGG19 yields network activations at conv4. Simultaneously, style image activations are obtained at earlier to middle convolution layers (conv1, conv2, conv3, conv4, conv5).

#### Gram Matrix Encoding: 
Transforming activations into Gram matrix representation serves as a descriptor for the image's 'style,' capturing its unique artistic characteristics.

#### Synthesis Objective: 
Our objective is to generate an output image that seamlessly merges the content of one image with the stylistic features of another.

#### Loss Calculations:

##### Content Loss: 
Quantified by the L2 distance between the content and generated images.
##### Style Loss: 
Computed as the sum of L2 distances between Gram matrices from different VGG19 layers for content and style images.
##### Total Variation Loss: 
Enhances spatial coherence and reduces noise in the generated image.
##### Total Loss: 
The cumulative sum of the above losses, each multiplied by its respective weight.
##### Optimization Technique: 
The L-BFGS iterative optimization method is employed to progressively minimize these losses, leading to the synthesis of the desired artistic results.


## Dependencies
### Python 3.9+
### Framework: PyTorch
### Libraries: os, numpy, cv2, matplotlib, torchvision
