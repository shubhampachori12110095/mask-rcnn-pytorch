import torch
import torch.nn as nn


class ClsBBoxHead_fc(nn.Module):
    def __init__(self, depth, pool_size, num_classes):
        super(ClsBBoxHead_fc, self).__init__()
        self.depth = depth
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_size)
        self.fc_0 = nn.Linear(depth, 1024)
        self.fc_1 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc_cls = nn.Linear(1024, num_classes)
        self.fc_bbox = nn.Linear(1024, num_classes * 4)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(-1, self.depth)
        x = self.fc_0(x)
        x = self.fc_1(x)

        fc_out_cls = self.fc_cls(x)
        fc_out_bbox = self.fc_bbox(x)
        prob_cls = self.log_softmax(fc_out_cls)
        prob_bbox = self.log_softmax(fc_out_bbox)

        return prob_cls, prob_bbox


class ClsBBoxHead_fcn(nn.Module):
    def __init__(self, depth, pool_size, num_classes):
        super(ClsBBoxHead_fcn, self).__init__()
        self.conv1 = nn.Conv2d(depth, 1024, kernel_size=pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc_cls = nn.Linear(1024, num_classes)
        self.fc_bbox = nn.Linear(1024, num_classes * 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(-1, 1024)
        fc_out_cls = self.fc_cls(x)
        prob_cls = self.softmax(fc_out_cls)
        reg_bbox = self.fc_bbox(x)

        return prob_cls, reg_bbox
