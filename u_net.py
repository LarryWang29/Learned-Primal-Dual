import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.conv1 = self.conv_block(in_channels, 64, 3, 1, 1)
        self.maxpool1 = self.maxpool_block(2, 2, 0)
        self.conv2 = self.conv_block(64, 128, 3, 1, 1)
        self.maxpool2 = self.maxpool_block(2, 2, 1)
        self.conv3 = self.conv_block(128, 256, 3, 1, 1)
        self.maxpool3 = self.maxpool_block(2, 2, 1)
        self.conv4 = self.conv_block(256, 512, 3, 1, 1)
        self.maxpool4 = self.maxpool_block(2, 2, 0)

        self.middle = self.conv_block(512, 1024, 3, 1, 1)

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
       maxpool = nn.Sequential(
                   nn.MaxPool2d(kernel_size, stride, padding)
       )
       return maxpool

    def transposed_block(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
       transposed = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
       return transposed

    def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
       final = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
       return final

    def forward(self, x):
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

