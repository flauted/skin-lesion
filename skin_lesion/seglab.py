# modified from https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/models/seg_net.py
import math

import torch
from torch import nn
from torch.nn import functional as F

from skin_lesion.models import vgg19_bn, resnet101


def initialize_weights(*models):
    for model in models:
        if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
            nn.init.kaiming_normal(model.weight)
            if model.bias is not None:
                model.bias.data.zero_()
        elif isinstance(model, nn.BatchNorm2d):
            model.weight.data.fill_(1)
            model.bias.data.zero_()
        else:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()


class Multiscale(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        ind_scale = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, ind_scale, (1, 1), stride=1,
                 padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(ind_scale),
            nn.ReLU(inplace=True))

        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=s, stride=1, padding=math.floor(s/2)) for s in (3, 5, 7)
        ])
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ind_scale, ind_scale, (3, 3), stride=1,
                    padding=d, dilation=d, groups=1, bias=True),
                nn.BatchNorm2d(ind_scale),
                nn.ReLU(inplace=True))
            for d in (3, 6, 12)
        ])
        initialize_weights(self.conv1, self.convs[0], self.convs[1], self.convs[2])

    def forward(self, input):
        x = self.conv1(input)
        return torch.cat([*[p(x) for p in self.pools],
                          *[c(x) for c in self.convs]], dim=1)


class Upsample(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        initialize_weights(self.conv1, self.conv2)

    def forward(self, multiscaled, feat_maps):
        x = F.interpolate(multiscaled, scale_factor=4, mode="bilinear")
        x = torch.cat((x, feat_maps), dim=1)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=4, mode="bilinear")
        x = self.conv2(x)
        return x


class SegLab(nn.Module):
    def __init__(self, num_classes, in_channels=3, pretrained=True, model_dir=None,
                 model="resnet101"):
        super(SegLab, self).__init__()
        if model == "resnet101":
            res = resnet101(pretrained=pretrained, model_dir=model_dir)
            if in_channels != res.conv1.in_channels:
                res.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.enc1 = nn.Sequential(
                res.conv1, res.bn1, res.relu, res.maxpool
            )
            self.enc2 = res.layer1
            self.enc3 = res.layer2
            self.enc4 = res.layer3
            self.enc5 = res.layer4

            self.conv1 = nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=1, padding=2, dilation=2)
            self.conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=2, dilation=2)
            self.conv3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=2, dilation=2)

            self.multiscale = Multiscale(512)
            self.upsample = Upsample(640, num_classes)

        else:
            vgg = vgg19_bn(pretrained=pretrained, model_dir=model_dir)
            features = list(vgg.features.children())
            if in_channels != features[0].in_channels:
                features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.enc1 = nn.Sequential(*features[0:7])
            self.enc2 = nn.Sequential(*features[7:14])
            self.enc3 = nn.Sequential(*features[14:27])
            self.enc4 = nn.Sequential(*features[27:40])
            self.enc5 = nn.Sequential(*features[40:])

            self.multiscale = Multiscale(512)
            self.upsample = Upsample(512, num_classes)


    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        x = self.conv1(enc4)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.multiscale(x)

        x = self.upsample(x, enc2)
        return x
