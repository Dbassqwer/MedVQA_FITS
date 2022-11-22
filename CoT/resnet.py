# print("PyTorch Version: ", torch.__version__)
# print("Torchvision Version: ", torchvision.__version__)

import torch.nn as nn
import torch.nn.functional as F

# from activations_jit import *
# from activations_me import *

from aggregation_zeropad import *

# from util.activations import *



def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        if stride > 1:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(places),
                # nn.ReLU(inplace=True),
                nn.AvgPool2d(3, 2, padding=1),
                CotLayer(places, kernel_size=3),
                # nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
                # nn.BatchNorm2d(places),
                # nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(places),
                # nn.ReLU(inplace=True),
                CotLayer(places, kernel_size=3),
                # nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
                # nn.BatchNorm2d(places),
                # nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion),
            )
        self.relu = nn.ReLU(inplace=True)
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )

    def forward(self, x):

        residual = x

        # print('bottle x',x.type())
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out2 = self.relu(out)
        out2 += residual

        return out2


class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        # self.conv1 = Conv1(in_planes=3, places=64)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.b1 = nn.BatchNorm2d(64)
        self.r1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('????????????????resnent,',x.type())
        x = self.conv1(x)
        # print('@#@$!#$!$$!$!')
        x = self.b1(x)
        x = self.r1(x)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet152():
    return ResNet([3, 8, 36, 3])


class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CotLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1,
                                           padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(dim)
        # act = get_act_layer('swish')
        # self.act = act(inplace=True)
        # pytorch < 1.7 没有silu，需要自己手写
        self.act = nn.SiLU(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix * dim, 1)
        )

    def forward(self, x):
        # print('cot laty:',x.type())
        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, 1, -1, self.kernel_size * self.kernel_size, qk_hh, qk_ww)

        x = self.conv1x1(x)
        x = self.local_conv(x, w)
        x = self.bn(x)
        x = self.act(x)

        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)

        return out.contiguous()


# def get_act_layer(name='relu'):
#     """ Activation Layer Factory
#     Fetching activation layers by name with this function allows export or torch script friendly
#     functions to be returned dynamically based on current config.
#     """
#     _has_silu = 'silu' in dir(torch.nn.functional)
#     _ACT_LAYER_ME = dict(
#         silu=nn.SiLU if _has_silu else SwishMe,
#         swish=nn.SiLU if _has_silu else SwishMe,
#         mish=MishMe,
#         hard_sigmoid=HardSigmoidMe,
#         hard_swish=HardSwishMe,
#         hard_mish=HardMishMe,
#     )
#     _ACT_LAYER_JIT = dict(
#         silu=nn.SiLU if _has_silu else SwishJit,
#         swish=nn.SiLU if _has_silu else SwishJit,
#         mish=MishJit,
#         hard_sigmoid=HardSigmoidJit,
#         hard_swish=HardSwishJit,
#         hard_mish=HardMishJit
#     )
#     _ACT_LAYER_DEFAULT = dict(
#         silu=nn.SiLU if _has_silu else Swish,
#         swish=nn.SiLU if _has_silu else Swish,
#         mish=Mish,
#         relu=nn.ReLU,
#         relu6=nn.ReLU6,
#         leaky_relu=nn.LeakyReLU,
#         elu=nn.ELU,
#         prelu=nn.PReLU,
#         celu=nn.CELU,
#         selu=nn.SELU,
#         gelu=nn.GELU,
#         sigmoid=Sigmoid,
#         tanh=Tanh,
#         hard_sigmoid=HardSigmoid,
#         hard_swish=HardSwish,
#         hard_mish=HardMish,
#     )
#     if not name:
#         return None
#     if not (is_no_jit() or is_exportable() or is_scriptable()):
#         if name in _ACT_LAYER_ME:
#             return _ACT_LAYER_ME[name]
#     if not is_no_jit():
#         if name in _ACT_LAYER_JIT:
#             return _ACT_LAYER_JIT[name]
#     return _ACT_LAYER_DEFAULT[name]
#
#
# def is_no_jit():
#     _NO_JIT = False
#     return _NO_JIT
#
#
# def is_exportable():
#     _EXPORTABLE = False
#     return _EXPORTABLE
#
#
# def is_scriptable():
#     _SCRIPTABLE = False
#     return _SCRIPTABLE


if __name__ == '__main__':
    # model = torchvision.models.resnet50()
    model = ResNet152().to('cuda')
    # print(model)
    # print(list(model.children()))
    # input = torch.randn(16,3,224,224).to('cuda')
    # out = model(input)
    # print(out.shape)
