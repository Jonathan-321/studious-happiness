#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
U-Net architecture implementation for layout segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution."""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """U-Net architecture for layout segmentation."""
    
    def __init__(self, n_channels=3, n_classes=9, bilinear=True, pretrained_encoder=True):
        """
        Initialize U-Net.
        
        Args:
            n_channels (int): Number of input channels
            n_classes (int): Number of output classes
            bilinear (bool): Whether to use bilinear upsampling
            pretrained_encoder (bool): Whether to use pretrained encoder
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Use ResNet34 as encoder
        if pretrained_encoder:
            resnet = models.resnet34(pretrained=True)
            self.inc = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu
            )
            self.down1 = nn.Sequential(
                resnet.maxpool,
                resnet.layer1
            )
            self.down2 = resnet.layer2
            self.down3 = resnet.layer3
            self.down4 = resnet.layer4
            
            # Adjust channel numbers for skip connections
            self.inc_channels = 64
            self.down1_channels = 64
            self.down2_channels = 128
            self.down3_channels = 256
            self.down4_channels = 512
        else:
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)
            
            # Channel numbers for skip connections
            self.inc_channels = 64
            self.down1_channels = 128
            self.down2_channels = 256
            self.down3_channels = 512
            self.down4_channels = 1024 // factor

        # Decoder
        self.up1 = Up(self.down4_channels + self.down3_channels, self.down3_channels, bilinear)
        self.up2 = Up(self.down3_channels + self.down2_channels, self.down2_channels, bilinear)
        self.up3 = Up(self.down2_channels + self.down1_channels, self.down1_channels, bilinear)
        self.up4 = Up(self.down1_channels + self.inc_channels, self.inc_channels, bilinear)
        self.outc = OutConv(self.inc_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits

class UNetWithResnet50Encoder(nn.Module):
    """U-Net with ResNet50 encoder for layout segmentation."""
    
    def __init__(self, n_channels=3, n_classes=9):
        """
        Initialize U-Net with ResNet50 encoder.
        
        Args:
            n_channels (int): Number of input channels
            n_classes (int): Number of output classes
        """
        super(UNetWithResnet50Encoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Encoder (ResNet50 layers)
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # 64 channels
        self.pool = resnet.maxpool
        self.encoder2 = resnet.layer1  # 256 channels
        self.encoder3 = resnet.layer2  # 512 channels
        self.encoder4 = resnet.layer3  # 1024 channels
        self.encoder5 = resnet.layer4  # 2048 channels
        
        # Decoder
        self.decoder1 = Up(2048 + 1024, 512)
        self.decoder2 = Up(512 + 512, 256)
        self.decoder3 = Up(256 + 256, 128)
        self.decoder4 = Up(128 + 64, 64)
        
        # Final output
        self.final = OutConv(64, n_classes)
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # 64 channels, 1/2 resolution
        p1 = self.pool(e1)     # 64 channels, 1/4 resolution
        e2 = self.encoder2(p1) # 256 channels, 1/4 resolution
        e3 = self.encoder3(e2) # 512 channels, 1/8 resolution
        e4 = self.encoder4(e3) # 1024 channels, 1/16 resolution
        e5 = self.encoder5(e4) # 2048 channels, 1/32 resolution
        
        # Decoder
        d1 = self.decoder1(e5, e4)  # 512 channels, 1/16 resolution
        d2 = self.decoder2(d1, e3)  # 256 channels, 1/8 resolution
        d3 = self.decoder3(d2, e2)  # 128 channels, 1/4 resolution
        d4 = self.decoder4(d3, e1)  # 64 channels, 1/2 resolution
        
        # Final output
        output = self.final(d4)  # n_classes channels, 1/2 resolution
        
        # Upsample to original size
        output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
        
        return output
