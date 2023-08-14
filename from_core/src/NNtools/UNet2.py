#based on: https://github.com/milesial/Pytorch-UNet/tree/master/unet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bnorm1=nn.BatchNorm2d(out_channels)
        self.relu1=nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu1(self.bnorm1(self.conv1(x)))

class NCBR(nn.Module):
    def __init__(self, in_channels, out_channels,N,skip=False,skipcat=False):
        super().__init__()
        assert N>1
        self.skip=skip
        self.skipcat=skipcat
        channels=[]
        channels.append(in_channels)
        for i in range(N):
            channels.append(out_channels)#len(channels) ==  N+1
        
        self.layers=nn.ModuleList()
        for i in range(N):
            self.layers.append(CBR(channels[i],channels[i+1]))
            
    def forward(self, x):
        for i,layer in enumerate(self.layers):
            if i==0:
                x=layer(x)
                x1=x
            else:
                x=layer(x)
        if self.skip:
            if self.skipcat:
                x=torch.cat([x,x1],dim=1)
            else:
                x=x+x1
        return x

class DownNCBR(nn.Module):
    """Downscaling with maxpool then NCBR"""
    def __init__(self, in_channels, out_channels,N,skip=False,skipcat=False):
        super().__init__()
        self.maxpool=nn.MaxPool2d(2)
        self.ncbr=NCBR(in_channels, out_channels,N=N,skip=skip,skipcat=skipcat)

    def forward(self, x):
        return self.ncbr(self.maxpool(x))

class UpNCBR(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels,N,skip=False,skipcat=False):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.ncbr = NCBR(in_channels, out_channels,N=N,skip=skip,skipcat=skipcat)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.ncbr(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=1):
        super(OutConv, self).__init__()
        padding=kernel_size//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,padding=padding)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,N=2,width=32,skip=False,skipcat=False,catorig=False,outker=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.catorig=catorig

        self.inc = NCBR(self.n_channels, 2*width,N=N,skip=skip)
        self.down1 = DownNCBR(2*width, 2*width if skipcat else 4*width,N=N,skip=skip,skipcat=skipcat)
        self.down2 = DownNCBR(4*width, 4*width if skipcat else 8*width,N=N,skip=skip,skipcat=skipcat)
        self.down3 = DownNCBR(8*width, 8*width if skipcat else 16*width,N=N,skip=skip,skipcat=skipcat)
        self.down4 = DownNCBR(16*width, 16*width if skipcat else 32*width,N=N,skip=skip,skipcat=skipcat)
        self.up1 = UpNCBR(32*width, 8*width if skipcat else 16*width,N=N,skip=skip,skipcat=skipcat)
        self.up2 = UpNCBR(16*width, 4*width if skipcat else 8*width,N=N,skip=skip,skipcat=skipcat)
        self.up3 = UpNCBR(8*width, 2*width if skipcat else 4*width,N=N,skip=skip,skipcat=skipcat)
        self.up4 = UpNCBR(4*width, width if skipcat else 2*width,N=N,skip=skip,skipcat=skipcat)
        if self.catorig:
            self.outc = OutConv(2*width+self.n_channels, self.n_classes,kernel_size=outker)
        else:
            self.outc = OutConv(2*width, self.n_classes,kernel_size=outker)

    def forward(self, x):
        orig=x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.catorig:
            logits=self.outc(torch.cat([x,orig],axis=1))
        else:
            logits=self.outc(x)
        return logits

class UNet_past(nn.Module):
    def __init__(self, n_channels, n_classes,N=2,width=64,skip=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = NCBR(self.n_channels, width,N=N,skip=skip)
        self.down1 = DownNCBR(width, 2*width,N=N,skip=skip)
        self.down2 = DownNCBR(2*width, 4*width,N=N,skip=skip)
        self.down3 = DownNCBR(4*width, 8*width,N=N,skip=skip)
        self.down4 = DownNCBR(8*width, 16*width,N=N,skip=skip)
        self.up1 = UpNCBR(16*width, 8*width,N=N)
        self.up2 = UpNCBR(8*width, 4*width,N=N)
        self.up3 = UpNCBR(4*width, 2*width,N=N)
        self.up4 = UpNCBR(2*width, width,N=N)
        self.outc = OutConv(width, self.n_classes)

    def forward(self, x):
        orig=x
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

class UNet_Den(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,init=0):
        super(UNet_Den, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = TripleConv(self.n_channels, 64)
        self.down1 = DownTriple(64, 128)
        self.down2 = DownTriple(128, 256)
        self.down3 = DownTriple(256, 512)
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
    
    