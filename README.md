# maskrcnn-pytorch

## (Currently, this project is under fast update state, will be fully available very soon.)

Mask R-CNN implementation in PyTorch

![maskrcnn-result](http://chuantu.biz/t6/250/1520606201x-1404795469.png)

## Usage
(Will be available very soon.)

```python
from maskrcnn import MaskRCNN

mask_rcnn = MaskRCNN(num_classes=1000)

def train():
    pass
def predict():
    pass
```

## Source directory explain

#### 1. backbone: 

Several backbone models support Mask R-CNN, like ResNet-101-FPN.

#### 2. proposal:

RoI(Region of Interest) Proposal, like RPN and variants.

#### 3. pooling:

Pooling for fixed dimensional representation, like RoIAlign and some variants.

#### 4. head:
Predict heads include classification head, bounding box head, mask head and their variants.

#### 5. lib:

Some third-party lib this project based on.


## Reference:

1. [Kaiming He et al. Mask R-CNN](https://arxiv.org/abs/1703.06870)
2. [Shaoqing Ren et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
3. [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
4. [ruotianluo/pytorch-faster-rcnn](ruotianluo/pytorch-faster-rcnn)
5. [TuSimple/mx-maskrcnn](https://github.com/TuSimple/mx-maskrcnn)
6. [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)