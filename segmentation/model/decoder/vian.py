# ------------------------------------------------------------------------------
# Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["VIAN"]


# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, dilation):
#         modules = [
#             nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         ]
#         super(ASPPConv, self).__init__(*modules)


# class ASPPPooling(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ASPPPooling, self).__init__()
#         self.aspp_pooling = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.ReLU()
#         )
#
#     def set_image_pooling(self, pool_size=None):
#         if pool_size is None:
#             self.aspp_pooling[0] = nn.AdaptiveAvgPool2d(1)
#         else:
#             self.aspp_pooling[0] = nn.AvgPool2d(kernel_size=pool_size, stride=1)
#
#     def forward(self, x):
#         size = x.shape[-2:]
#         x = self.aspp_pooling(x)
#         # return F.interpolate(x, size=size, mode='bilinear', align_corners=True)
#         return F.interpolate(x, size=size)


class VIAN(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(VIAN, self).__init__()
        # out_channels = 256
        modules = []
        # modules.append(nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU()))
        #
        # rate1, rate2, rate3 = tuple(atrous_rates)
        # modules.append(ASPPConv(in_channels, out_channels, rate1))
        # modules.append(ASPPConv(in_channels, out_channels, rate2))
        # modules.append(ASPPConv(in_channels, out_channels, rate3))
        # modules.append(ASPPPooling(in_channels, out_channels))
        self.version = 1
        if self.version == 1:
            modules.append(nn.Sequential(
                     nn.Conv2d(in_channels, out_channels//2, kernel_size=3, padding=1, bias=False),
                     nn.BatchNorm2d(out_channels//2),
                     nn.ReLU()))
            modules.append(nn.Sequential(
                     nn.Conv2d(out_channels//2, out_channels//4, kernel_size=3, padding=1, bias=False),
                     nn.BatchNorm2d(out_channels//4),
                     nn.ReLU()))
            modules.append(nn.Sequential(
                     nn.Conv2d(out_channels//4, out_channels//8, kernel_size=3, padding=1, bias=False),
                     nn.BatchNorm2d(out_channels//8),
                     nn.ReLU()))
            modules.append(nn.Sequential(
                     nn.Conv2d(out_channels//8, out_channels//16, kernel_size=3, padding=1, bias=False),
                     nn.BatchNorm2d(out_channels//16),
                     nn.ReLU()))
            modules.append(nn.Sequential(
                     nn.Conv2d(out_channels//16, out_channels - out_channels//2 - out_channels//4 - out_channels//8 - out_channels//16, kernel_size=3, padding=1, bias=False),
                     nn.BatchNorm2d(out_channels - out_channels//2 - out_channels//4 - out_channels//8 - out_channels//16),
                     nn.ReLU()))
        elif self.version == 2:
            modules.append(nn.Sequential(
                     nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                     nn.BatchNorm2d(out_channels),
                     nn.ReLU()))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def set_image_pooling(self, pool_size):
        pass # self.convs[-1].set_image_pooling(pool_size)

    def forward(self, x):
        y = None
        res = []
        for conv in self.convs:
            y = conv(x if y is None else y)
            res.append(y)
        res = torch.cat(res, dim=1)
        return self.project(res)
