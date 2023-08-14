#based on: https://github.com/milesial/Pytorch-UNet/tree/master/unet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1=nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bnorm1=nn.BatchNorm2d(mid_channels)
        self.relu1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bnorm2=nn.BatchNorm2d(out_channels)
        self.relu2=nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu2(self.bnorm2(self.conv2(self.relu1(self.bnorm1(self.conv1(x))))))

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool=nn.MaxPool2d(2)
        self.dc=DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.dc(self.maxpool(x))

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.bilinear=bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.dc = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.dc = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.up(x1)
        else:
            x1 = self.upconv(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.dc(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,init=0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)
        
        if init==0:
            pass
        elif init==1:
            for name, param in self.named_parameters():
                if "conv" in name and 'weight' in name:
                    n = param.size(0) * param.size(2) * param.size(3)
                    param.data.normal_().mul_(np.sqrt(2. / n))
                    #print(name)
                elif "conv" in name and 'bias' in name:
                    param.data.fill_(0)
                    #print(name)
                elif "bnorm" in name and 'weight' in name:
                    param.data.fill_(1)
                    #print(name)
                elif "bnorm" in name and 'bias' in name:
                    param.data.fill_(0)
                    #print(name)
                else:
                    pass
                    #print("no init",name)

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
    
    