#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of BiSeNet v2:
    
Reference:
    BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation
    Changqian Yu, Changxin Gao, Jingbo Wang, Gang Yu, Chunhua Shen, Nong Sang
    arXiv:2004.02147 (https://arxiv.org/abs/2004.02147)
"""


import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict

class ConvBn(nn.Module):
    """
    2D convolution followed by a batch normalization
    Arguments as in nn.Conv2d
    Padding is automatically adjusted to SAME size
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):     
        super().__init__()
        if isinstance(kernel_size,tuple):
            padding = tuple([(c-1)//2 for c in kernel_size])
        else:
            padding = (kernel_size-1)//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, bias=False, groups=groups)
        self.batchnorm = nn.BatchNorm2d(out_channels, affine=True)
        
    def forward(self,x):
        return self.batchnorm(self.conv(x))

class ConvBnReLU(ConvBn):
    """
    2D convolution followed by a batch normalization and ReLU activation
    Arguments as in nn.Conv2d
    Padding is automatically adjusted to SAME size
    """
    def forward(self,x):
        return F.relu(self.batchnorm(self.conv(x)),inplace=True)

class StemBlock(nn.Module):
    """
    Stem Block
    Args:
        in_channels
        out_channels
    """
    def __init__(self,
             in_channels=3,
             out_channels=16):
        super().__init__()
        C = out_channels
        self.conv_in = ConvBnReLU(in_channels, C, kernel_size=3, stride=2)
        self.conv_branch = nn.Sequential(
            ConvBnReLU(C, C//2, kernel_size=1, stride=1),
            ConvBnReLU(C//2, C, kernel_size=3, stride=2),
            )
        self.conv_out = ConvBnReLU(2*C, C, kernel_size=3, stride=1)
        
    def forward(self,x):
        x = self.conv_in(x)
        x = torch.cat([self.conv_branch(x),
                    F.max_pool2d(x, kernel_size=3, stride=2,padding=1),
                    ],dim=1)
        return self.conv_out(x)

class GatherAndExpand(nn.Module):
    """
    Gather and Expand block
    Args:
        C: number of channels
        K: expansion factor
    """
    def __init__(self, C, K=6):
        super().__init__()
        self.main = nn.Sequential(OrderedDict([
          ('conv1', ConvBnReLU(C, C, kernel_size=3)),
          ('conv2_space', ConvBn(C,K*C, kernel_size=3, groups=C)),
          ('conv2_depth', ConvBn(K*C, C, kernel_size=1)),
        ]))
        self.skip = nn.Identity()
            
    def forward(self,x):
        return F.relu(self.main(x)+self.skip(x),inplace=True)

class GatherAndExpandStride(nn.Module):
    """
    Gather and Expand block with stride 2
    Args:
        C: number of channels
        K: expansion factor
    """
    def __init__(self, in_channels, out_channels, K=6):
        super().__init__()
        C = in_channels
        self.main = nn.Sequential(OrderedDict([
          ('conv1', ConvBnReLU(C, C, kernel_size=3)),
          ('conv2_space', ConvBn(C,K*C, kernel_size=3, groups=C, stride=2)),
          ('conv3_space', ConvBn(K*C,K*C, kernel_size=3, groups=K*C)),
          ('conv3_depth', ConvBn(K*C, out_channels, kernel_size=1)),
        ]))
        self.skip = nn.Sequential(OrderedDict([
          ('conv_skip_space', ConvBn(C, C, kernel_size=3, stride=2, groups=C)),
          ('conv_skip_depth', ConvBn(C, out_channels, kernel_size=1)),
        ]))
        
    def forward(self,x):
        return F.relu(self.main(x)+self.skip(x),inplace=True)

class ContextEmbedding(nn.Module):
    def __init__(self,n_channels):
        super().__init__()
        C = n_channels
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(C),
            ConvBnReLU(C, C, 1, 1))
        self.conv = nn.Conv2d(C, C, 3, 1, padding=1)
    def forward(self, x):
         return self.conv(x + self.block(x))
class SegmentationHead(nn.Module):
    """
    Segmentation head
    Args:
        in_channels: number of channels
        n_hidden: number of channels after 3x3 convolution
        n_classes: number of classes
        scale_factor: scale factor for upsampling
    """
    def __init__(self, in_channels, n_hidden, n_classes):
        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_hidden,3)
        self.conv2 = nn.Conv2d(n_hidden, n_classes, kernel_size=1)
        
    def forward(self, x, size):
        z = self.conv2(self.conv1(x))
        return F.interpolate(z, size=size, mode="bilinear", align_corners=True)

class Aggregate(nn.Module):
    """
    Cd: number of channels in details branch
    Cs: number of channels in semantic branch
    """
    def __init__(self,Cd,Cs):
        super().__init__()
        self.detail_1 = nn.Sequential(
            ConvBn(Cd, Cd, 3, 1, groups=Cd),
            nn.Conv2d(Cd, Cd, 1, 1))
        self.detail_2 = nn.Sequential(
            ConvBn(Cd, Cd, 3, 2),
            nn.AvgPool2d(3, 2, padding=1))
        self.semantic_1 = ConvBn(Cs, Cd, 3, 1)
        self.semantic_2 = nn.Sequential(
            ConvBn(Cs, Cs, 3, 1, groups=Cs),
            nn.Conv2d(Cs, Cd, 1, 1),
            nn.Sigmoid())
        self.conv_out = ConvBn(Cd, Cd, 3, 1)
        
    def forward(self,x_detail, x_semantic):
        upsample_op = lambda x: F.interpolate(x,size=x_detail.shape[2:],
                                              mode="bilinear", align_corners=True)
        x_semantic_1 = self.semantic_1(x_semantic)
        x_semantic_1 = torch.sigmoid(upsample_op(x_semantic_1))
        x_1 = self.detail_1(x_detail) * x_semantic_1
        x_2 = self.detail_2(x_detail) * self.semantic_2(x_semantic)
        return self.conv_out(x_1 + upsample_op(x_2))
    
class BiSeNetv2(nn.Module):
    def __init__(self,in_channels, n_classes,
                 seg_hidden=256, K=6, ratio = 4):
        super().__init__()
        self.details_branch = nn.Sequential(
            ConvBnReLU(3, 64, 3, stride=2),
            ConvBnReLU(64, 64, 3),
            ConvBnReLU(64, 64, 3, stride=2),
            ConvBnReLU(64, 64, 3),
            ConvBnReLU(64, 64, 3),
            ConvBnReLU(64, 128, 3, stride=2),
            ConvBnReLU(128, 128, 3),
            ConvBnReLU(128, 128, 3),
            )
        s_base = 64//ratio
        self.semantic_branch = nn.ModuleList([
            StemBlock(in_channels,s_base),
            nn.Sequential(
                GatherAndExpandStride(s_base, s_base*2, K),
                GatherAndExpand(s_base*2, K),
                ),
            nn.Sequential(
                GatherAndExpandStride(s_base*2, s_base*4, K),
                GatherAndExpand(s_base*4, K),
                ),
            nn.Sequential(
                GatherAndExpandStride(s_base*4, s_base*8, K),
                GatherAndExpand(s_base*8, K),        
                GatherAndExpand(s_base*8, K),     
                GatherAndExpand(s_base*8, K),     
                ),
            ContextEmbedding(s_base*8),
            ])
        self.agg = Aggregate(128, s_base*8)
        self.head = SegmentationHead(128,seg_hidden,n_classes)
        
        self.AuxTowers = nn.ModuleList([
                SegmentationHead(s_base,seg_hidden,n_classes),
                SegmentationHead(2*s_base,seg_hidden,n_classes),
                SegmentationHead(4*s_base,seg_hidden,n_classes),
                SegmentationHead(8*s_base,seg_hidden,n_classes),
                SegmentationHead(8*s_base,seg_hidden,n_classes),
            ])
        
    def forward(self,x, aux_towers=None):
        input_size = x.shape[2:]
        x_details = self.details_branch(x)
        x_semantic_list = []
        x_semantic = x
        for ge_layers in self.semantic_branch:
            x_semantic = ge_layers(x_semantic)
            x_semantic_list.append(x_semantic)
        x = self.agg(x_details, x_semantic_list[-1])
        x_output = self.head(x, input_size)
        
        if aux_towers is None:
            return x_output
        else:
            # only 4 aux heads, the output of the context embedding layer
            # is discarded
            extra = [self.AuxTowers[ii](x_semantic_list[ii], input_size) for ii in aux_towers]
            return (x_output, *extra)
            