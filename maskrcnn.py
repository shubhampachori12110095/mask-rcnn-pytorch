from backbone.resnet_101_fpn import ResNet_101_FPN
from proposal.rpn import RPN
from head.cls_bbox import ClsBBoxHead_fc as ClsBBoxHead
from head.mask import MaskHead
from pooling.roi_align import RoiAlign

import torch
import torch.nn.functional as F
import torch.nn as nn


class MaskRCNN(nn.Module):
    """Mask R-CNN
    
    References: https://arxiv.org/pdf/1703.06870.pdf
    
    Notes: In comments below, we assume N: batch size, M: number of roi,
        C: feature map channel, H: image height, W: image width
    """

    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()
        self.num_classes = num_classes
        self.fpn = ResNet_101_FPN()
        self.rpn = RPN(dim=512)
        self.roi_align = RoiAlign(grid_size=(14, 14))
        self.cls_box_head = ClsBBoxHead(depth=512, pool_size=14, num_classes=num_classes)
        self.mask_head = MaskHead(depth=512, pool_size=14, num_classes=num_classes)

    def forward(self, x, gt_classes=None, gt_bboxes=None, gt_masks=None):
        """
        Args:
            x: image data. NxCxHxW.  
            gt_classes: NxMx1, ground truth class ids.
            gt_bboxes: NxMx4(x1, y1, x2, y2), ground truth bounding boxes.
            gt_masks: NxMxHxW, ground truth masks.
        Returns:
            prob_cls: NxMx(num_classes), probability of classification. 
            reg_bbox: NxMx(x1, y1, x2, y2), regression of bounding-box. 
            prob_cls:  NxMx2(num_classes), probability of mask.
        """
        if self.training:
            assert gt_classes is not None
            assert gt_bboxes is not None
            assert gt_masks is not None

        p2, p3, p4, p5, p6 = self.fpn(x)
        rpn_features_rpn = [p2, p3, p4, p5, p6]
        fpn_features = [p2, p3, p4, p5]
        rois, rpn_loss_cls, rpn_loss_bbox = self.rpn(rpn_features_rpn, gt_bboxes)
        roi_pools = self._roi_align_fpn(fpn_features, rois, img_height=x.size(2),
                                        img_width=x.size(3))
        roi_pools = torch.cat(roi_pools, 0)
        prob_cls, reg_bbox = self.cls_box_head(roi_pools)
        prob_mask = self.mask_head(roi_pools)

        # reshape back to NxMx(num_classes)
        prob_cls = prob_cls.view(x.size(0), -1, self.num_classes)
        reg_bbox = reg_bbox.view(x.size(0), -1, self.num_classes)
        prob_mask = prob_mask.view(x.size(0), -1, self.num_classes)
        mask_loss = rpn_loss_cls + rpn_loss_bbox
        return prob_cls, reg_bbox, prob_mask, mask_loss

    def _genarate_target(self):
        pass

    def _calc_maskrcnn_loss(self, prob_cls, reg_bbox, prob_mask, gt_classes,
                           gt_bboxes, gt_masks):
        cls_loss = F.nll_loss(prob_cls, gt_classes)
        bbox_loss = F.smooth_l1_loss(reg_bbox, gt_bboxes)
        mask_loss = F.binary_cross_entropy(prob_mask, gt_masks)
        maskrcnn_loss = cls_loss + bbox_loss + mask_loss
        return maskrcnn_loss

    def _roi_align_fpn(self, fpn_features, rois, img_width, img_height):
        """When use fpn backbone, set RoiAlign use different levels of fpn feature pyramid
         according to RoI size.
         
        Args:
            fpn_features: [p2, p3, p4, p5], 
            rois: NxMx5(n ,x1, y1, x2, y2), RPN proposals.
            img_width: Input image width.
            img_height: Input image height.

        Returns:
            roi_pools: RoI after use RoIAlign.
        """
        # flatten NxMx4 to (NxM)x4
        rois_reshape = rois.view(-1, rois.size(-1))
        bboxes = rois_reshape[:, 1:]
        bbox_indexes = rois_reshape[:, 0]
        roi_pools = []
        for idx, bbox in enumerate(bboxes):
            # In feature pyramid network paper, alpha is 224 and image short side 800 pixels,
            # for using of small image input, like maybe short side 256, here alpha is
            # parameterized by image short side size.
            alpha = 224 // 800 * (img_width if img_width <= img_height else img_height)
            bbox_width = torch.abs(bbox[0] - bbox[2])
            bbox_height = torch.abs(bbox[1] - bbox[3])
            log2 = torch.log(torch.sqrt(bbox_height * bbox_width)) / torch.log(2) / alpha
            level = torch.floor(4 + log2) - 2  # minus 2 make level 0 indexed
            # Rois small or big enough may get level below 0 or above 3.
            level = torch.clamp(level, 0, 3)
            bbox = torch.unsqueeze(bbox, 0)
            roi_pool_per_box = self.roi_align(fpn_features[level], bbox, bbox_indexes[idx])
            roi_pools.append(roi_pool_per_box)
        return roi_pools
