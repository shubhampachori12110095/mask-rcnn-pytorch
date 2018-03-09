from backbone.resnet_101_fpn import ResNet_101_FPN
from proposal.rpn import RPN
from head.cls_bbox import ClsBBoxHead_fc as ClsBBoxHead
from head.mask import MaskHead
from pooling.roi_align import RoiAlign

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
        p2, p3, p4, p5, p6 = self.fpn(x)
        fpn_features_rpn = [p2, p3, p4, p5, p6]
        fpn_features = [p2, p3, p4, p5]

        for feature in fpn_features_rpn:
            rois, rpn_loss_cls, rpn_loss_bbox = self.rpn(feature)
        # flatten NxMx5 to (NxM)x5
        rois_reshape = rois.view(-1, rois.size()[-1])
        bboxes = rois_reshape[:, 1:]
        bbox_indexes = rois_reshape[:, 0]
        roi_pools = self._roi_align_fpn(fpn_features_rpn, bboxes, bbox_indexes,
                                        img_height=x.size()[2], img_width=x.size()[3])
        roi_pools = torch.cat(roi_pools, 0)
        prob_cls, reg_bbox = self.cls_box_head(roi_pools)
        prob_mask = self.mask_head(roi_pools)
        # reshape back to NxMx(num_classes)
        prob_cls = prob_cls.view(x.size()[0], -1, self.num_classes)
        reg_bbox = reg_bbox.view(x.size()[0], -1, self.num_classes)
        prob_mask = prob_mask.view(x.size()[0], -1, self.num_classes)
        return prob_cls, reg_bbox, prob_mask

    def _roi_align_fpn(self, fpn_features, bboxes, bbox_indexes, img_width, img_height):
        """When use fpn backbone, set RoiAlign use different levels of fpn feature pyramid
         according to RoI size.
         
        Args:
            fpn_features: [p2, p3, p4, p5]
            bboxes: Bounding boxes.
            bbox_indexes: Indexes of bounding box in mini-batch.
            img_width: Input image width.
            img_height: Input image height.

        Returns:
            roi_pools: RoI after use RoIAlign.
        """
        roi_pools = []
        for idx, bbox in enumerate(bboxes):
            # set alpha parameterized by image short side size, for
            # consideration of small image input.
            alpha = 224 // 800 * (img_width if img_width <= img_height else img_height)
            bbox_width = torch.abs(bbox[0] - bbox[2])
            bbox_height = torch.abs(bbox[1] - bbox[3])
            log2 = torch.log(torch.sqrt(bbox_height * bbox_width)) / torch.log(2) / alpha
            level = torch.floor(4 + log2) - 2  # minus 2 make level 0 indexed
            # rois small enough may make level below zero，set these rois to use p2 feature
            level = level if level >= 0 else 0
            bbox = torch.unsqueeze(bbox, 0)
            roi_pool_per_box = self.roi_align(fpn_features[level], bbox, bbox_indexes[idx])
            roi_pools.append(roi_pool_per_box)
        return roi_pools
