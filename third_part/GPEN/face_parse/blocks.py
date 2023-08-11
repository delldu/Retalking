# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
import pdb

class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """
    def __init__(self, channels, normalize_shape=None, norm_type='bn', ref_channels=None):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        # elif norm_type == 'pixel':
        #     self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            # self.norm = lambda x: x*1.0
            self.norm = nn.Identity() # for torch.jit.script
        else:
            assert 1==0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x, ref=None):
        # if self.norm_type == 'spade':
        #     return self.norm(x, ref)
        # else:
        #     return self.norm(x)
        return self.norm(x)


class ReluLayer(nn.Module):
    """Relu Layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu 
            - SELU
            - none: direct pass
    """
    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            # self.func = lambda x: x*1.0
            self.func = nn.Identity() # for torch.jit.script            
        else:
            assert 1==0, 'Relu type {} not support.'.format(relu_type)
        # pdb.set_trace()

    def forward(self, x):
        return self.func(x) # xxxx8888, torch.jit.script()


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale='none', norm_type='none', 
        relu_type='none', bias=True):
        super(ConvLayer, self).__init__()

        self.norm_type = norm_type
        if norm_type in ['bn']:
            bias = False
        
        stride = 2 if scale == 'down' else 1

        # self.scale_func = lambda x: x
        self.scale_func = nn.Identity() #  for torch.jit.script
        self.use_up = False
        if scale == 'up':
            self.use_up = True
            # self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')

        self.reflection_pad = nn.ReflectionPad2d(int(np.ceil((kernel_size - 1.)/2))) 
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.relu = ReluLayer(out_channels, relu_type)
        self.norm = NormLayer(out_channels, norm_type=norm_type)

        # pdb.set_trace()
        # torch.jit.script(self) ==> error !!!, xxxx8888

    def forward(self, x):
        out = x
        if self.use_up:
            out = F.interpolate(out, scale_factor=2.0, mode='nearest')
        out = self.reflection_pad(out)
        out = self.conv2d(out)
        out = self.norm(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    """
    Residual block recommended in: http://torch.ch/blog/2016/02/04/resnets.html
    """
    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none'):
        super(ResidualBlock, self).__init__()
        if scale == 'none' and c_in == c_out:
            # self.shortcut_func = lambda x: x
            self.shortcut_func = nn.Identity() # for torch.jit.script
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)
        
        scale_config_dict = {'down': ['none', 'down'], 'up': ['up', 'none'], 'none': ['none', 'none']}
        scale_conf = scale_config_dict[scale]

        self.conv1 = ConvLayer(c_in, c_out, 3, scale_conf[0], norm_type=norm_type, relu_type=relu_type) 
        self.conv2 = ConvLayer(c_out, c_out, 3, scale_conf[1], norm_type=norm_type, relu_type='none')
  
    def forward(self, x):
        identity = self.shortcut_func(x)

        res = self.conv1(x)
        res = self.conv2(res)
        return identity + res
        

