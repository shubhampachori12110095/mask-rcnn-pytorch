from backbone.resnet_101_fpn import ResNet_101_FPN
from proposal.rpn import RPN
from head.cls_bbox import ClsBBoxHead_fc as ClsBBoxHead
from head.mask import MaskHead
from lib.nms.pth_nms import pth_nms as nms
from pool.roi_align import RoiAlign

import torch
import torch.nn as nn


class MaskRCNN(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.fpn = ResNet_101_FPN()
        self.rpn = RPN(dim=512)
        self.roi_align = RoiAlign(grid_size=(14, 14))
        self.cls_box_head = ClsBBoxHead(depth=512, pool_size=14, num_classes=num_classes)
        self.mask_head = MaskHead(depth=512, pool_size=14, num_classes=num_classes)

    def forward(self, x):
        """
        
        Args:
            x: image data. NxCxHxW  

        Returns:
            prob_cls: NxMx(num_classes), probability of classification. 
            reg_bbox: NxMx(x1, y1, x2, y2), regression of bounding-box. 
            prob_cls:  NxMx2(num_classes), probability of mask.
        """
        p2, p3, p4, p5 = self.fpn(x)
        fpn_features = [p2, p3, p4, p5]
        for feature in fpn_features:
            rois, rpn_loss_cls, rpn_loss_bbox = self.rpn(feature)
        # flatten NxMx5 to (NxM)x5
        rois_reshape = rois.view(-1, rois.size()[-1])
        bboxs = rois_reshape[:, 1:]
        bbox_idx = rois_reshape[:, 0]
        roi_pool = self.roi_align(feature, bboxs, bbox_idx)
        prob_cls, reg_bbox = self.cls_box_head(roi_pool)
        prob_mask = self.mask_head(roi_pool)
        # reshape back to NxMx(num_classes)
        prob_cls = prob_cls.view(x.size()[0], -1, self.num_classes)
        reg_bbox = reg_bbox.view(x.size()[0], -1, self.num_classes)
        prob_mask = prob_mask.view(x.size()[0], -1, self.num_classes)
        return prob_cls, reg_bbox, prob_mask
