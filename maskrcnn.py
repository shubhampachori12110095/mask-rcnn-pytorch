from backbone.resnet_101_fpn import ResNet_101_FPN
from proposal.rpn import RPN
from head.box import BoxHead
from head.cls import ClsHead
from head.mask import MaskHead
from lib.nms.pth_nms import pth_nms as nms
from lib.roi_align.roi_align import RoIAlign

import torch
import torch.nn as nn


class MaskRCNN(nn.Module):
    def __init__(self):
        self.fpn = ResNet_101_FPN()
        self.rpn = RPN()
        self.roi_align = RoIAlign()
        self.box_head = BoxHead()
        self.cls_head = ClsHead()
        self.mask_head = MaskHead()

    def forward(self, x):
        p2, p3, p4, p5 = self.fpn(x)
        fpn_features = [p2, p3, p4, p5]
        for feature in fpn_features:
            rpn_result = self.rpn(feature)
        if not self.training:
            nms_result = nms(rpn_result)

        roi_align_result = self.roi_align(nms_result)

        box_result = self.box_head(roi_align_result)
        cls_result = self.cls_head(roi_align_result)
        mask_result = self.mask_head(roi_align_result)

        return box_result, cls_result, mask_result
