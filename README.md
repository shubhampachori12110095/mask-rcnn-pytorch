# maskrcnn-pytorch

## English:

## (Currently, this project is in continuous update state, will be available soon.)

Mask R-CNN implementation in PyTorch

## usage
(need update)

```python
from maskrcnn import MaskRCNN

mask_rcnn = MaskRCNN(num_classes=1000)

def train():
    pass
def predict():
    pass
```

## source directory

#### backbone: 

Several backbones model support Mask R-CNN, like ResNet-101-FPN.

#### proposal:

RoI(Regions of interest) Proposal, like RPN and variants.

#### pool:

Pooling for fixed dimensional representation, like RoIAlign and some variants.

#### lib:

Some third party code this project based on.


Reference:
1. [Kaiming He et al. Mask R-CNN](https://arxiv.org/abs/1703.06870)
2. [Shaoqing Ren et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
3. [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
4. [ruotianluo/pytorch-faster-rcnn](ruotianluo/pytorch-faster-rcnn)
5. [TuSimple/mx-maskrcnn](https://github.com/TuSimple/mx-maskrcnn)
6. [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)

---

## 中文：

## (本项目在持续更新中, 很快将可以使用。)

Mask R-CNN的PyTorch实现

参考:
1. [Kaiming He et al. Mask R-CNN](https://arxiv.org/abs/1703.06870)
2. [Shaoqing Ren et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
3. [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
4. [ruotianluo/pytorch-faster-rcnn](ruotianluo/pytorch-faster-rcnn)
5. [TuSimple/mx-maskrcnn](https://github.com/TuSimple/mx-maskrcnn)
6. [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)