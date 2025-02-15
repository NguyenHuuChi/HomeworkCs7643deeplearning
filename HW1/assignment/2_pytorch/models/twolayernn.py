import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TwoLayerNN(nn.Module):
    def __init__(self, im_size, hidden_dim, n_classes):
        '''
        Create components of a two layer neural net classifier (often
        referred to as an MLP) and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            n_classes (int): Number of classes to score
        '''
        super(TwoLayerNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.channels, self.height, self.width = im_size
        self.input_dim = self.channels *self.height* self.width
        self.output_dim = n_classes
        self.hidden_dim =hidden_dim
        self.fc1= nn.Linear(self.input_dim,self.hidden_dim, bias=True)
        self.fc2= nn.Linear(self.hidden_dim, self.output_dim, bias = True)
        self.relu = nn.ReLU()

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the NN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take very few lines of code.
        #############################################################################
        N,C,H,W= images.shape
        images=torch.reshape(images,(N,-1))
        x=self.fc1(images)
        x=self.relu(x)
        scores=self.fc2(x)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

