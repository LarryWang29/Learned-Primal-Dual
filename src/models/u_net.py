"""
This module contains the Pytorch implementation of an adjusted version of the U-Net
architecture. The original U-Net architecture was proposed in the paper "U-Net: Convolutional
Networks for Biomedical Image Segmentation" by Ronneberger et al (https://arxiv.org/abs/1505.04597).
The adjusted version of the U-Net architecture is used to perform image denoising, and changes
include the addition of a skip connection between the input and output of the network and slight
alterations to MaxPool2d blocks.
"""

import torch
import torch.nn as nn


class UNet(nn.Module):
    """
      This class implements the adjusted U-Net architecture for image denoising. The architecture
      consists of an encoder-decoder network with skip connections between the encoder and decoder
      layers. The encoder consists of four convolutional blocks with maxpool layers, and the decoder
      consists of four transposed convolutional blocks. The middle layer consists of a single
      convolutional block. The final layer is a single convolutional layer that outputs the denoised
      image. The skip connections are implemented by concatenating the output of the encoder layers
      with the input of the corresponding decoder layers. The channel sizes are 64, 128, 256, 512, and
      1024.
    """
    def __init__(self, in_channels=1, out_channels=1):
        """
        Initalisation function for the U-Net architecture.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.    
        """
        super(UNet, self).__init__()
        # Down-sampling part of the network with four convolutional blocks
        self.conv1 = self.conv_block(in_channels, 64, 3, 1, 1)
        self.maxpool1 = self.maxpool_block(2, 2, 0)
        self.conv2 = self.conv_block(64, 128, 3, 1, 1)
        self.maxpool2 = self.maxpool_block(2, 2, 1)
        self.conv3 = self.conv_block(128, 256, 3, 1, 1)
        self.maxpool3 = self.maxpool_block(2, 2, 1)
        self.conv4 = self.conv_block(256, 512, 3, 1, 1)
        self.maxpool4 = self.maxpool_block(2, 2, 0)

        # Middle part of the network with a single convolutional block
        self.middle = self.conv_block(512, 1024, 3, 1, 1)

        # Up-sampling part of the network with four transposed convolutional blocks
        self.upsample4 = self.transposed_block(1024, 512, 3, 2, 1, 1)
        self.upconv4 = self.conv_block(1024, 512, 3, 1, 1)
        self.upsample3 = self.transposed_block(512, 256, 3, 2, 1, 1)
        self.upconv3 = self.conv_block(512, 256, 3, 1, 1)
        self.upsample2 = self.transposed_block(256, 128, 3, 2, 1, 1)
        self.upconv2 = self.conv_block(256, 128, 3, 1, 1)
        self.upsample1 = self.transposed_block(128, 64, 3, 2, 1, 1)
        self.upconv1 = self.conv_block(128, 64, 3, 1, 1)

        self.final = self.final_layer(64, out_channels, 1, 1, 0)

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
       """
       The convolutional block consists of two convolutional layers with batch normalization 
       and ReLU activation functions.

       Parameters
       ----------
       in_channels : int
            The number of input channels.
       out_channels : int
            The number of output channels.
       kernel_size : int
            The size of the convolutional kernel.
       stride : int
            The stride of the convolution.
       padding : int
            The padding of the convolution.
       Returns
       -------
       convolution : torch.nn.Sequential
            The convolutional block.
       """
       convolution = nn.Sequential(
                     nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                     nn.BatchNorm2d(out_channels),
                     nn.ReLU(),
                     nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                     nn.BatchNorm2d(out_channels),
                     nn.ReLU(),
                    )

       return convolution

    def maxpool_block(self, kernel_size, stride, padding):
       """
       The maxpool block consists of a maxpool layer, which reduces the spatial dimensions of the
       input tensor.

       Parameters
       ----------
       kernel_size : int
            The size of the maxpool kernel.
       stride : int
            The stride of the maxpool.
       padding : int
            The padding of the maxpool.
       
       Returns
       -------
       maxpool : torch.nn.Sequential
            The maxpool block.
       """
       maxpool = nn.Sequential(
                   nn.MaxPool2d(kernel_size, stride, padding)
       )
       return maxpool

    def transposed_block(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
       """
       The transpose block consists of a transposed convolutional layer, which increases the spatial
       dimensions of the input tensor.

       Parameters
       ----------
       in_channels : int
            The number of input channels.
       out_channels : int
            The number of output channels.
       kernel_size : int
            The size of the transposed convolutional kernel.
       stride : int
            The stride of the transposed convolution.
       padding : int
            The padding of the transposed convolution.
       output_padding : int
            The output padding of the transposed convolution.
       
       Returns
       -------
       transposed : torch.nn.Sequential
            The transposed convolutional block.
       """
       transposed = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
       return transposed

    def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
       """
       The final layer consists of a single convolutional layer that outputs the denoised image.

       Parameters
       ----------
       in_channels : int
            The number of input channels.
       out_channels : int
            The number of output channels.
       kernel_size : int
            The size of the convolutional kernel.
       stride : int
            The stride of the convolution.
       padding : int
            The padding of the convolution.
       
       Returns
       -------
       final : torch.nn.Conv2d
            The final convolutional layer.
       """
       final = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
       return final

    def forward(self, x):
        """
        The forward pass of the entire U-Net architecture.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the network.
        
        Returns
        -------
        x : torch.Tensor
            The denoised image.
        """
        # downsampling part
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # middle part
        middle = self.middle(maxpool4)

        # upsampling part
        upsample4 = self.upsample4(middle)
        upconv4 = self.upconv4(torch.cat([conv4, upsample4], 1))
        upsample3 = self.upsample3(upconv4)
        upsample3 = upsample3[:, :, :-1, :-1]
        upconv3 = self.upconv3(torch.cat([conv3, upsample3], 1))
        upsample2 = self.upsample2(upconv3)
        upsample2 = upsample2[:, :, :-1, :-1]
        upconv2 = self.upconv2(torch.cat([conv2, upsample2], 1))
        upsample1 = self.upsample1(upconv2)
        upconv1 = self.upconv1(torch.cat([conv1, upsample1], 1))

        final_layer = self.final(upconv1)

        return x + final_layer

