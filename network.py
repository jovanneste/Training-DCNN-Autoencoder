import numpy as np
import torch
import torch.nn.functional as F

class ConvolutionalAutoencoder(torch.nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.conv_2 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.deconv_1 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.deconv_2 = torch.nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 7 * 7, 10)



    def classification(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


    def forward(self, x):
        x = self.conv_1(x)
        x = F.leaky_relu(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = F.leaky_relu(x)
        x = self.pool_2(x)

        ### LATENT SPACE CLASSIFICATION
        classification_output = self.classification(x)

        x = self.deconv_1(x)
        x = F.leaky_relu(x)
        x = self.deconv_2(x)
        x = F.leaky_relu(x)
        logits = x[:, :, 2:30, 2:30]
        probas = torch.sigmoid(logits)
        return logits, probas, classification_output


