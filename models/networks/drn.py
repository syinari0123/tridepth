# BSD 3-Clause License
#
# Copyright (c) 2017, Fisher Yu
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation[0], bias=False, dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation[1], bias=False, dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        # currently not compatible with running on CPU
        self.weights = torch.autograd.Variable(
            torch.zeros(num_channels, 1, stride, stride).cuda()
        )
        self.weights[:, :, 0, 0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)


class UpProjModule(nn.Module):
    # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
    #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
    #   bottom branch: 5*5 conv -> batchnorm

    def __init__(self, in_channels, out_channels, up_flag=True):
        super(UpProjModule, self).__init__()
        self.up_flag = up_flag
        self.unpool = Unpool(in_channels)
        self.upper_branch = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels,
                                kernel_size=5, stride=1, padding=2, bias=False)),
            ('batchnorm1', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU()),
            ('conv2', nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, stride=1, padding=1, bias=False)),
            ('batchnorm2', nn.BatchNorm2d(out_channels)),
        ]))
        self.bottom_branch = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels,
                               kernel_size=5, stride=1, padding=2, bias=False)),
            ('batchnorm', nn.BatchNorm2d(out_channels)),
        ]))
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.up_flag:
            x = self.unpool(x)
        x1 = self.upper_branch(x)
        x2 = self.bottom_branch(x)
        x = x1 + x2
        x = self.relu(x)
        return x


class DRN_d_54(nn.Module):
    """
    [Reference]
    https://github.com/fyu/drn/blob/master/drn.py
    """

    def __init__(self, block=Bottleneck,
                 layers=[1, 1, 3, 4, 6, 3, 1, 1],
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 model_type="upconv",
                 feat_size=512,
                 pretrained=True):
        super(DRN_d_54, self).__init__()
        self.inplanes = channels[0]
        self.out_dim = channels[-1]
        self.model_type = model_type

        # layer definition
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_conv_layers(channels[0], layers[0], stride=1)
        self.layer2 = self._make_conv_layers(channels[1], layers[1], stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4],
                                       dilation=2, new_level=False)
        self.layer6 = self._make_layer(block, channels[5], layers[5],
                                       dilation=4, new_level=False)
        self.layer7 = self._make_conv_layers(channels[6], layers[6],
                                             dilation=2)
        self.layer8 = self._make_conv_layers(channels[7], layers[7],
                                             dilation=1)

        # Concat hierarchical features
        if self.model_type == "simple":
            self.conv1 = conv(16, feat_size // 4)
            self.conv2 = conv(32, feat_size // 4)
            self.conv3 = conv(256, feat_size // 4)
            self.conv4 = conv(512, feat_size // 4)

        elif self.model_type == "upconv":
            self.upproj4 = UpProjModule(512, 512, up_flag=True)
            self.upproj3 = UpProjModule(512 + 256, 256, up_flag=True)
            self.upproj2 = UpProjModule(256 + 32, 32, up_flag=True)
            self.upproj1 = UpProjModule(32 + 16, 16, up_flag=False)

            self.conv4 = conv(512, feat_size // 4)
            self.conv3 = conv(256, feat_size // 4)
            self.conv2 = conv(32, feat_size // 4)
            self.conv1 = conv(16, feat_size // 4)

        else:
            raise NotImplementedError

        if pretrained:
            drn_d_54_url = "http://dl.yf.io/drn/drn_d_54-0e0534ff.pth"
            self.load_weights(drn_d_54_url)

    def load_weights(self, weight_url):
        pretrained_dict = model_zoo.load_url(weight_url)
        model_dict = self.state_dict()

        # Filter out unnecessary keys
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in model_dict and (v.shape == model_dict[k].shape)
        }
        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # Load the new state dict
        self.load_state_dict(model_dict)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()

        x = self.layer0(x)

        # C1
        x1 = self.layer1(x)  # [B,16,228,304]

        # C2
        x2 = self.layer2(x1)  # [B,32,114,152]

        # C3
        x3 = self.layer3(x2)  # [B,256,57,76]

        # C4-C8
        x3_tmp = self.layer4(x3)
        x3_tmp = self.layer5(x3_tmp)
        x3_tmp = self.layer6(x3_tmp)
        x3_tmp = self.layer7(x3_tmp)
        x4 = self.layer8(x3_tmp)  # [B,512,29,30]

        # decoder
        if self.model_type == "simple":
            # interpolate
            x1 = F.interpolate(x1, size=x.size()[-2:])
            x2 = F.interpolate(x2, size=x.size()[-2:])
            x3 = F.interpolate(x3, size=x.size()[-2:])
            x4 = F.interpolate(x4, size=x.size()[-2:])
            # conv
            x1 = self.conv1(x1)
            x2 = self.conv2(x2)
            x3 = self.conv3(x3)
            x4 = self.conv4(x4)
            # concat
            out = torch.cat((x1, x2, x3, x4), 1)  # [B,512,H,W]
            return out

        elif self.model_type == "upconv":
            x4_up = self.upproj4(x4)
            x4_up = F.interpolate(x4_up, size=x3.size()[-2:])

            x3_up = self.upproj3(torch.cat((x4_up, x3), 1))
            x3_up = F.interpolate(x3_up, size=x2.size()[-2:])

            x2_up = self.upproj2(torch.cat((x3_up, x2), 1))
            x2_up = F.interpolate(x2_up, size=x1.size()[-2:])

            x1_up = self.upproj1(torch.cat((x2_up, x1), 1))

            # Interpolate
            x1_up = self.conv1(F.interpolate(x1_up, size=x.size()[-2:]))
            x2_up = self.conv2(F.interpolate(x2_up, size=x.size()[-2:]))
            x3_up = self.conv3(F.interpolate(x3_up, size=x.size()[-2:]))
            x4_up = self.conv4(F.interpolate(x4_up, size=x.size()[-2:]))

            out = torch.cat((x1_up, x2_up, x3_up, x4_up), 1)  # [B,512,H,W]
            return out

        else:
            raise NotImplementedError
