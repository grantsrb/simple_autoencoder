# Simple Autoencoder

## Description
This project consists of a simple autoencoder. Here is a high level description of what the autoencoder does:

The autoencoder first reduces images from Cifar-10 to a low dimensional vector of random variables using a conv net. 

More specifically, the autoencoder reduces the image into two vectors. One is treated as a vector of means for a normal distribution. The other is treated as a corresponding vector of standard deviations. These two vectors are used to construct a z vector whose entries are drawn from the normal distributions created by the corresponding mean and standard deviation vectors. 

The z vector is used to predict the class of the image and is then fed into a convolution transpose net to reconstruct the original image.

Losses from both the prediction error and the reconstruction error are used in backprop.

Ideally, the autoencoder can be trained to create a semantically rich z vector that can be used for activities like Reinforcement Learning or machine memory. These applications are not included in this project.

## Experimental Setup
The observations used for this autoencoder were synthesized from the Cifar-10 image dataset. 

