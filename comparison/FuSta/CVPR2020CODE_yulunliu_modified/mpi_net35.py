import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def conv1_1(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv1_2(in_channels,out_channels,kernel=4,stride=2,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv2_1(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv2_2(in_channels,out_channels,kernel=4,stride=2,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv3_1(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv3_2(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv3_3(in_channels,out_channels,kernel=4,stride=2,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv4_1(in_channels,out_channels,kernel=3,stride=1,dilate=2,padding=2):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    dilation=dilate,
    bias=True,
    groups=1)

def conv4_2(in_channels,out_channels,kernel=3,stride=1,dilate=2,padding=2):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    dilation=dilate,
    bias=True,
    groups=1)

def conv4_3(in_channels,out_channels,kernel=3,stride=1,dilate=2,padding=2):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    dilation=dilate,
    bias=True,
    groups=1)

def conv5_1(in_channels,out_channels,kernel=4,stride=2,padding=1):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel,
        stride=stride,
        padding=padding)

def conv5_2(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv5_3(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv6_1(in_channels,out_channels,kernel=4,stride=2,padding=1):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel,
        stride=stride,
        padding=padding)

def conv6_2(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv7_1(in_channels,out_channels,kernel=4,stride=2,padding=1):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel,
        stride=stride,
        padding=padding)

def conv7_2(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv7_3(in_channels,out_channels,kernel=1,stride=1,dilate=1,padding=0):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)


class mpinet(nn.Module):
    def __init__(self, in_channels=60, out_channels=38,
                 start_filts=64):
        super(mpinet, self).__init__()
        self.out_channels=out_channels
        self.conv11=conv1_1(in_channels,start_filts*2)
        self.conv12=conv1_2(start_filts*2,start_filts*4)
        self.conv21=conv2_1(start_filts*4,start_filts*4)
        self.conv22=conv2_2(start_filts*4,start_filts*8)
        self.conv31=conv3_1(start_filts*8,start_filts*8)
        self.conv32=conv3_2(start_filts*8,start_filts*8)
        self.conv33=conv3_3(start_filts*8,start_filts*16)
        self.conv41=conv4_1(start_filts*16,start_filts*16)
        self.conv42=conv4_2(start_filts*16,start_filts*16)
        self.conv43=conv4_3(start_filts*16,start_filts*16)

        self.conv51=conv5_1(start_filts*32,start_filts*8)
        self.conv52=conv5_2(start_filts*8,start_filts*8)
        self.conv53=conv5_3(start_filts*8,start_filts*8)
        self.conv61=conv6_1(start_filts*16,start_filts*4)
        self.conv62=conv6_2(start_filts*4,start_filts*4)
        self.conv71=conv7_1(start_filts*8,start_filts*2)
        self.conv72=conv7_2(start_filts*2,start_filts*2)
        self.conv73=conv7_3(start_filts*2,out_channels)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x1):
        
        out1_1=self.conv11(x1)
        out1_2=self.conv12(out1_1)
        out2_1=self.conv21(out1_2)
        out2_2=self.conv22(out2_1)
        out3_1=self.conv31(out2_2)
        out3_2=self.conv32(out3_1)
        out3_3=self.conv33(out3_2)
        out4_1=self.conv41(out3_3)
        out4_2=self.conv42(out4_1)
        out4_3=self.conv43(out4_2)
        
        out5_1=self.conv51(torch.cat((out4_3,out3_3),1))
        out5_2=self.conv52(out5_1)
        out5_3=self.conv53(out5_2)
        out6_1=self.conv61(torch.cat((out5_3,out2_2),1))
        out6_2=self.conv62(out6_1)
        out7_1=self.conv71(torch.cat((out6_2,out1_2),1))
        out7_2=self.conv72(out7_1)
        out7_3=self.conv73(out7_2)
        
        return out7_3