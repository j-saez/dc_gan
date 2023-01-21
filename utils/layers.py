import torch.nn as nn

"""
    Name: conv_block
    Description: 
        
        Returns a convolutional block for the discriminator or a transpose convolutional block for the generator.
        It is up to the developer to add batch normalization and activation to the block.

    Inputs:

        >> in_ch: (int) quantity of channels that the block will receive as inputs
        >> out_ch: (int) quantity of channels that the block will ouput
        >> kernel_size: (int) kernel_size for the convolution or transpose convolution
        >> stride: (int) stride to be added during the convolution or transpose convolution.
        >> padding: (int) padding to be added during the convolution or transpose convolution.
        >> norm: (bool) Whether to add batch normalization to the block or not.
        >> activation: (bool) Whether to add LeakyReLU activation or not.
        >> discriminator: (bool) Whether the block is for the discriminator or generator.

    Outputs:

        >> layer: (nn.Module) Layer for the selected model (discriminator/generator).
"""
def conv_block(in_ch,out_ch,kernel_size,stride,padding,norm,activation,discriminator):
    bias = False if norm else True # If batch norm bias is not needed
    layer = nn.Sequential()
    if discriminator:
        layer.append(nn.Conv2d(in_ch, out_ch, kernel_size,stride,padding,bias))
    else: # Generator
        layer.append(nn.ConvTranspose2d(in_ch,out_ch,kernel_size,stride,padding,bias))
    if norm:
        layer.append(nn.BatchNorm2d(out_ch))
    if activation:
        layer.append(nn.LeakyReLU(0.2))
    return layer
