
# Face Mask Detection using CNN




## Overview

This project is a deep learning-based face mask detection system using a Convolutional Neural Network (CNN). The model is trained to classify images into two categories:

-With Mask (Label: 1)

-Without Mask (Label: 0)
## Dataset

The dataset consists of labeled images upto 7553:

1 for images where a person is wearing a mask.

0 for images where a person is not wearing a mask.
## Preprocessing Steps

1.Shuffle Dataset: Used random.shuffle() to randomize the order of images.

2.Assign Variables:

X: Stores image data (both masked and unmasked images).

Y: Stores corresponding labels.

3.Train-Test Split:

The dataset was split into training and testing sets using train_test_split().

4.Normalization:

Images were scaled to normalize pixel values for better training performance.
## CNN Model

This code defines a Convolutional Neural Network (CNN) using Keras for a binary classification task (num_of_classes = 2). Here's a brief explanation:

-Input Layer: The model takes input images of size 128x128 with 3 color channels (RGB).

-Convolutional Layers: Two Conv2D layers with 32 filters, 3x3 kernel size, and ReLU activation extract features from the input images.

-Pooling Layers: Two MaxPooling2D layers with 2x2 pool size reduce the spatial dimensions, helping to reduce computation and overfitting.

-Flatten Layer: The Flatten layer converts the 2D feature maps into a 1D vector for the fully connected layers.

-Dense Layers: Two fully connected (Dense) layers with 128 units and ReLU activation, followed by Dropout layers (with a 0.5 dropout rate) to prevent overfitting.

-Output Layer: A Dense layer with num_of_classes (2) units and sigmoid activation for binary classification.
## Results

The model is capable of predicting whether an image contains a mask or not, demonstrating its ability to learn but does not yet achieve high accuracy, suggesting potential for further optimization.Reason may be train on small dataset
## Future Enhancements

1.Data Augmentation: Implement data augmentation techniques such as rotation, flipping, and zooming to increase dataset variability and improve model generalization.

2.Pre-trained Models: Use pre-trained models like MobileNet, ResNet, or VGG16 to improve feature extraction and increase accuracy through transfer learning.
## Special Thanks

This project was inspired by the tutorial from the siddhardhan Channel on YouTube.

Video:DL Project 5. Face Mask Detection using Convolutional Neural Network (CNN) - Deep Learning Projects

Channel: https://www.youtube.com/@Siddhardhan
