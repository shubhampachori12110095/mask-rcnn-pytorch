import torch.nn as nn
from lib.model.rpn.rpn import _RPN


class RPN(nn.Module):
    """Region Proposal Network encapsulation.   
    """
    def __init__(self, dim):
        """
        Args: 
            dim: depth of input feature map, e.g., 512
        """
        super(RPN, self).__init__()
        self.rpn = _RPN(dim)

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        """
        
        Args:
            base_feat: 
            im_info: 
            gt_boxes: 
            num_boxes: 
        Returns:
             rois: NxMx5(n, x1, y1, x2, y2)
                N: batch size. M: number of roi after nms. n: bbox idx in batch  
             rpn_loss_cls: 
             rpn_loss_bbox:
        """
        return self.rpn(base_feat, im_info, gt_boxes, num_boxes)
