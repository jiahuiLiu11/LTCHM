import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer
from thop import profile


class DynamicConv2d(nn.Module):  ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=1,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            ConvModule(dim,
                       dim // reduction_ratio,
                       kernel_size=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'), ),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1), )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):

        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K // 2,
                     groups=B * C,
                     bias=bias)

        return x.reshape(B, C, H, W)


class ResBlock(nn.Module):
    def __init__(self, num_filters=128):
        super(ResBlock, self).__init__()
        self.conv1 = DynamicConv2d(dim=num_filters, kernel_size=1, num_groups=2)
        self.relu1 = nn.ReLU()
        self.conv2 = DynamicConv2d(dim=num_filters, kernel_size=3, num_groups=2)
        self.relu2 = nn.ReLU()
        self.conv3 = DynamicConv2d(dim=num_filters, kernel_size=1, num_groups=2)

    def forward(self, x):
        res = self.relu1(self.conv1(x))
        res = self.relu2(self.conv2(res))
        res = self.conv3(res)
        res += x
        return res


class Attention(nn.Module):
    def __init__(self, num_filters=128):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.trunk_ResBlock1 = ResBlock(num_filters)
        self.trunk_ResBlock2 = ResBlock(num_filters)
        self.trunk_ResBlock3 = ResBlock(num_filters)
        self.attention_ResBlock1 = ResBlock(num_filters)
        self.attention_ResBlock2 = ResBlock(num_filters)
        self.attention_ResBlock3 = ResBlock(num_filters)

    def forward(self, x):
        trunk_branch = self.trunk_ResBlock1(x)
        trunk_branch = self.trunk_ResBlock2(trunk_branch)
        trunk_branch = self.trunk_ResBlock3(trunk_branch)

        attention_branch = self.attention_ResBlock1(x)
        attention_branch = self.attention_ResBlock2(attention_branch)
        attention_branch = self.attention_ResBlock3(attention_branch)
        attention_branch = self.conv1(attention_branch)
        attention_branch = self.sigmoid(attention_branch)

        # print("x.shape: ", x.shape)
        # print("attention.shape: ", attention_branch.shape)
        # print("trunk_branch.shape: ", trunk_branch.shape)
        result = x + torch.mul(attention_branch, trunk_branch)
        return result


# print('==> Building model..')
# # # 创建动态卷积模块
# # dynamic_conv = DynamicConv2d(dim=128, kernel_size=3, num_groups=2)
# attention = Attention(num_filters=192)
# #
# # 随机生成输入张量
# input_tensor = torch.randn(8, 192, 128, 128)
# out_put = attention(input_tensor)
# flops, params = profile(attention, (input_tensor,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
#
# # 应用动态卷积模块
# output_tensor = attention(input_tensor)
#
# print("输入张量形状:", input_tensor.shape)
# print("输出张量形状:", output_tensor.shape)