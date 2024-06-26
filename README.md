# Training-DCNN-Autoencoder


This network combines an autoencoder with a fully connected layer to simultaneously learn a reconstruction and classify images from the FMNIST dataset. In this way, we can use the latent space representation from the autoencoder to classify images and improve accuracy by forcing the network to learn good encodings. To update the weights of this network, we can use either the classifier loss (between the predicted output and the correct label), the encoder loss (between the reconstructed image and the input) or some joint loss. We demonstrate that using a joint loss improves model classification accuracy.
