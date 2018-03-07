import torch
import torch.nn as nn
import math


class MaskHead(nn.Module):
    def __init__(self, depth, pool_size, num_classes):
        super(MaskHead, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(256)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(padding=_calc_same_padding(x.size()))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(padding=_calc_same_padding(x.size()))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(padding=_calc_same_padding(x.size()))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(padding=_calc_same_padding(x.size()))
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)

        return x


def _calc_same_padding(input_size, kernel_size=3, stride=1):
    in_width = input_size[2]
    in_height = input_size[3]
    out_width = math.ceil(float(in_width) / float(stride))
    out_height = math.ceil(float(in_height) / float(stride))
    pad_width = ((out_width - 1) * stride + kernel_size - in_width) // 2
    pad_height = ((out_height - 1) * stride + kernel_size - in_height) // 2

    return pad_height, pad_width
