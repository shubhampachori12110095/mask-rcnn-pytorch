import torch.nn as nn
from lib.roi_align.roi_align import RoIAlign as _RoiAlign


class RoiAlign(nn.Module):
    """RoiAlign wrapper
    
    """
    def __init__(self, grid_size):
        """
        Args:
            grid_size(tuple): grid pool size apply to roi, e.g., (14, 14).  
        """
        super(RoiAlign, self).__init__()
        assert isinstance(grid_size, tuple)
        self.roi_align = _RoiAlign(crop_width=grid_size[0], crop_height=grid_size[1])

    def forward(self, feature_map, boxes, box_idx):
        """
        
        Args:
            feature_map: NxCxHxW
            boxes: Mx4 float box with (x1, y1, x2, y2) **without normalization**
            box_idx: M

        Returns:
            roi_pool: MxCxoHxoW  M: number of roi in all mini-batch.
        """
        roi_pool = self.roi_align(feature_map, boxes, box_idx)

        return roi_pool
