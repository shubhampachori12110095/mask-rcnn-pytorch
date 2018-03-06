import torch.nn as nn
from lib.model.rpn.rpn import _RPN


class RPN(nn.Module):
    """ Region Proposal Network 封装   
    """
    def __init__(self, dim):
        """
        Arg: 
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
             rois: 
             rpn_loss_cls: 
             rpn_loss_box:
        """
        return self.rpn(base_feat, im_info, gt_boxes, num_boxes)
