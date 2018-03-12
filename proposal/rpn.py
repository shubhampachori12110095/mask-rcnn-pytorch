import torch
import torch.nn as nn
from lib.model.rpn.rpn import _RPN
from lib.model.rpn.anchor_target_layer import _AnchorTargetLayer
from lib.model.rpn.proposal_layer import _ProposalLayer
from lib.model.rpn.config import cfg
from lib.nms.pth_nms import pth_nms as nms


class RPN(nn.Module):
    """Region Proposal Network Wrapper.   
    """

    def __init__(self, dim):
        """
        Args: 
            dim: depth of input feature map, e.g., 512
        """
        super(RPN, self).__init__()
        self.rpn = _RPN(dim)
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.anchor_scales = [8, 16, 32, 64, 128]
        self.feat_strides = [4, 8, 16, 32, 32]
        self.RPN_anchor_targets = [_AnchorTargetLayer(feat_stride=self.feat_strides[idx],
                                                      scales=scale,
                                                      ratios=self.anchor_ratios)
                                   for idx, scale in enumerate(self.anchor_scales)]
        self.RPN_proposals = [_ProposalLayer(feat_stride=self.feat_strides[idx],
                                             scales=scale,
                                             ratios=self.anchor_ratios)
                              for idx, scale in enumerate(self.anchor_scales)]

    def forward(self, feature_maps, gt_bboxes=None, im_info=None):
        """
        
        Args:
            feature_maps: [p2, p3, p4, p5, p6]
            gt_bboxes: 
            im_info: 
        Returns:
             rois: NxMx5(idx, x1, y1, x2, y2)
                N: batch size, M: number of roi after nms, idx: bbox index in mini-batch.
             rpn_loss_cls: Classification loss
             rpn_loss_bbox: Bounding box regression loss
        """
        batch_size = feature_maps.size(0)
        nms_output_num = cfg.TEST.RPN_POST_NMS_TOP_N
        if self.training:
            nms_output_num = cfg.TRAIN.RPN_POST_NMS_TOP_N
        rois_pre_nms = []
        rpn_loss_cls = 0
        rpn_loss_bbox = 0
        for idx, feature in enumerate(feature_maps):
            self.rpn.RPN_anchor_target = self.RPN_anchor_targets[idx]
            self.rpn.RPN_proposal = self.RPN_proposals[idx]
            rpn_single_result = self.rpn(feature, im_info, gt_bboxes, None)
            roi_single, loss_cls_single, loss_bbox_single = rpn_single_result
            rpn_loss_cls += loss_cls_single
            rpn_loss_bbox += loss_bbox_single
            roi_score = roi_single[:, :, 1]
            roi_bbox = roi_single[:, :, 2:]
            rois_pre_nms.append(torch.cat((roi_bbox, roi_score), 1))
        # NxMx5. torch.cat() at axis M.
        rois_pre_nms = torch.cat(rois_pre_nms, 1)
        rois = torch.zeros(batch_size, nms_output_num, 5)
        for i in range(batch_size):
            keep_idx = nms(rois_pre_nms[i], cfg.RPN_NMS_THRESH)
            keep_idx = keep_idx[:nms_output_num]
            rois_single = torch.cat([rois_pre_nms[i][idx] for idx in keep_idx])
            rois[i, :, 0] = i
            rois[i, :, 1:] = rois_single[:, :4]   # remove roi_score
        return rois, rpn_loss_cls, rpn_loss_bbox
