"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.builder import get_builder
from args import args

class wide_basic(nn.Module):
    def __init__(self, builder, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = builder.batchnorm(in_planes)
        self.conv1 = builder.conv3x3(in_planes, planes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=stride)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, planes, stride=stride),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, builder, depth, widen_factor, dropout_rate=0, num_classes=10):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        #print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = builder.conv3x3(3, nStages[0], stride=1, first_layer=True)
        self.layer1 = self._wide_layer(builder, wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(builder, wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(builder, wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = builder.batchnorm(nStages[3])
        self.linear = builder.conv1x1(nStages[3], num_classes)

    def _wide_layer(self, builder, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(builder, self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        # out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out.flatten(1)

def cWideResNet28_10():
    return Wide_ResNet(get_builder(), 28, 10, dropout_rate=0, num_classes=10)

def c100WideResNet28_10():
    return Wide_ResNet(get_builder(), 28, 10, dropout_rate=0, num_classes=100)
