import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ResNet, constant_init, kaiming_init, normal_init, xavier_init
from mmcv.runner import load_checkpoint

from mmdet.utils import get_root_logger
from ..builder import BACKBONES


class ConvBnReluLayer(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, stride, bias=False):
        super(ConvBnReluLayer, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding, 
                               stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

@BACKBONES.register_module()
class SSDResNet(ResNet):
    """ResNet Backbone network for single-shot-detection.

    Args:
        input_size (int): width and height of input, from {300, 512}.
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        out_indices (Sequence[int]): Output from which stages.

    Example:
        >>> self = SSDResNet(input_size=300, depth=181)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 300, 300)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 1024, 19, 19)
        (1, 512, 10, 10)
        (1, 256, 5, 5)
        (1, 256, 3, 3)
        (1, 256, 1, 1)
    """
    extra_setting = {
        (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256), # TODO
    }

    def __init__(self,
                 input_size,
                 depth,
                 out_indices=(0, 1, 2, 3),
                 l2_norm_scale=20.):
        # TODO: in_channels for mmcv.ResNet
        super(SSDResNet, self).__init__(
            depth=depth,
            out_indices=out_indices)
        self.input_size = input_size

        self.inplanes = 1024
        self.extra_layers = self._make_extra_layers(self.extra_setting)
        self.l2_norm = L2Norm(
            1, # todo
            l2_norm_scale)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        constant_init(self.l2_norm, self.l2_norm.scale)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out38x38 = x
        x = self.layer3(x)
        x = self.layer4(x)
        out19x19 = x

        out10x10, out5x5, out3x3, out1x1 = self.extra_layers(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return out38x38, out19x19, out10x10, out5x5, out3x3, out1x1

    def _make_extra_layers(self, outplanes):
        layers = []
        layers.append(ConvBnReluLayer(self.inplanes, 256, kernel_size=1, padding=0, stride=1))
        self.convbnrelu1_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)
        self.convbnrelu2_1 = ConvBnReluLayer(512, 256, kernel_size=1, padding=0, stride=1)
        self.convbnrelu2_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2) 
        self.convbnrelu3_1 = ConvBnReluLayer(512, 256, kernel_size=1, padding=0, stride=1)
        self.convbnrelu3_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)
        self.avgpool = nn.AvgPool2d(3, stride=1)

        return nn.Sequential(*layers)

class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        """L2 normalization layer.

        Args:
            n_dims (int): Number of dimensions to be normalized
            scale (float, optional): Defaults to 20..
            eps (float, optional): Used to avoid division by zero.
                Defaults to 1e-10.
        """
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        """Forward function."""
        # normalization layer convert to FP32 in FP16 training
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) *
                x_float / norm).type_as(x)
