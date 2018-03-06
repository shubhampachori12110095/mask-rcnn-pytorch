import torch.nn as nn
from lib.model.fpn import FPN101


class ResNet_101_FPN(nn.Module):
    """ ResNet 101 FPN 封装
    
    """
    def __init__(self):
        super(ResNet_101_FPN, self).__init__()
        self.fpn = FPN101()

    def forward(self, x):
        """
        Args:
            x: input image 
        Returns:
            p2:
            p3:
            p4:
            p5:
        """
        return self.fpn(x)
