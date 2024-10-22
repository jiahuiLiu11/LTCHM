from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import math

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]


class SaELayer(nn.Module):
    def __init__(self, in_channel, reduction=32):
        super(SaELayer, self).__init__()
        assert in_channel>=reduction and in_channel%reduction==0,'invalid in_channel in SaElayer'
        self.reduction = reduction
        self.cardinality=4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #cardinality 1
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel,in_channel//self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 2
        self.fc2 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 3
        self.fc3 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 4
        self.fc4 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_channel//self.reduction*self.cardinality, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y3 = self.fc3(y)
        y4 = self.fc4(y)
        y_concate = torch.cat([y1,y2,y3,y4],dim=1)
        y_ex_dim = self.fc(y_concate).view(b,c,1,1)

        return x * y_ex_dim.expand_as(x)


class LKA_AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # depth-wise convolution
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # depth-wise dilation convolution
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # channel convolution (1×1 convolution)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


class MS_FFN(nn.Module):
    def __init__(self, dim, scale=4, drop=0):
        super(MS_FFN, self).__init__()
        self.split = dim
        self.fc1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, stride=1, padding=0)

        self.dw_conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=dim)
        self.dw_conv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.dw_conv5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.dw_conv7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)

        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Conv2d(4 * dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        fc1 = self.fc1(x)
        fc1 = self.leaky_relu(fc1)
        conv1_x, conv3_x, conv5_x,conv7_x = torch.split(fc1,(self.split, self.split, self.split, self.split), dim=1)
        dw_conv1 = self.dw_conv1(conv1_x)
        # dw_conv1 = self.leaky_relu(dw_conv1)
        conv3_x = dw_conv1 + conv3_x
        dw_conv3 = self.dw_conv3(conv3_x)
        # dw_conv3 = self.leaky_relu(dw_conv3)
        conv5_x = conv5_x + dw_conv3
        dw_conv5 = self.dw_conv5(conv5_x)
        # dw_conv5 = self.leaky_relu(dw_conv5)
        conv7_x = dw_conv5 + conv7_x
        dw_conv7 = self.dw_conv7(conv7_x)
        # dw_conv7 = self.leaky_relu(dw_conv7)
        out = torch.cat([dw_conv1, dw_conv3, dw_conv5, dw_conv7], dim=1)
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = out + x

        return out

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim + input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.SaE = SaELayer(in_channel=input_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim + input_dim)
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, 4 * input_dim),
        #     nn.GELU(),
        #     nn.Linear(4 * input_dim, output_dim),
        # )
        self.mlp = MS_FFN(dim=2 * input_dim)
        self.scconv = ScConv(op_channel=input_dim)
        self.conv1_1 = nn.Conv2d(self.input_dim + self.input_dim, self.input_dim + self.input_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.input_dim + self.input_dim, self.input_dim + self.input_dim, 1, 1, 0, bias=True)
        # self.spa_cha_1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
        # self.leaky_relu = nn.LeakyReLU()
        # self.spa_cha_2 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
        self.proj = nn.Sequential(
            nn.Conv2d(self.input_dim + self.input_dim, self.input_dim + self.input_dim, kernel_size=3, padding=1, groups=self.input_dim + self.input_dim),
            nn.LeakyReLU(),
            nn.Conv2d(self.input_dim + self.input_dim, (self.input_dim + self.input_dim) // 4, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d((self.input_dim + self.input_dim) // 4, self.input_dim + self.input_dim, kernel_size=1), )

    def forward(self, x):
        conv_x = self.conv1_1(x)
        default = conv_x

        split_x = Rearrange('b c h w -> b h w c')(conv_x)
        split_x = self.ln1(split_x)
        split_x = Rearrange('b h w c -> b c h w')(split_x)

        # split_x = self.conv1_1(conv_x)
        sae_x, trans_x = torch.split(split_x, (self.input_dim, self.input_dim), dim=1)
        sae_x = self.SaE(sae_x)
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.msa(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        concat_x = torch.cat((sae_x, trans_x), dim=1)
        # concat_x = self.proj(concat_x) + concat_x
        concat_x = self.conv1_2(concat_x)
        concat_x = concat_x + default
        # x = x + self.drop_path(self.msa(self.ln1(x)))
        # x = x + self.drop_path(self.mlp(self.ln2(x)))

        default_2 = concat_x
        concat_x = Rearrange('b c h w -> b h w c')(concat_x)
        concat_x = self.ln2(concat_x)
        x_out2 = Rearrange('b h w c -> b c h w')(concat_x)
        # x_out2 = x_out1 + self.drop_path(self.mlp(self.ln2(x_out1)))
        # x_out1 = Rearrange('b h w c -> b c h w')(x_out1)
        x_out2 = self.mlp(x_out2)
        # x_out2 = Rearrange('b c h w -> b h w c')(x_out2)
        x_out2 = self.drop_path(x_out2) + default_2
        return x_out2

class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W'):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']
        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)
        # self.spa = SpatialAttention(in_channels=conv_dim)
        # self.cha = SaELayer(in_channel=conv_dim)
        # self.lka = LKA_AttentionModule(dim=conv_dim)
        # self.spa_cha_1 = nn.Conv2d(conv_dim, conv_dim, kernel_size=1, stride=1, padding=0)
        # self.leaky_relu = nn.LeakyReLU()
        # self.spa_cha_2 = nn.Conv2d(conv_dim, conv_dim, kernel_size=1, stride=1, padding=0)
        # self.sigmoid = nn.Sigmoid()
        # self.scconv = ScConv(op_channel=self.conv_dim)

    def forward(self, x):
        # conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        # conv_x = self.scconv(conv_x) + conv_x
        # conv_x = self.conv_block(conv_x) + conv_x
        # conv_x = self.cha(conv_x) + conv_x
        # trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(x)
        # trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        # res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        res = self.conv1_2(trans_x)
        x = x + res
        return x


class Hyper_analysis(nn.Module):
    def __init__(self, num_filters=128):
        super(Hyper_analysis, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.leaky_relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(num_filters, 2 * num_filters, 3, stride=2, padding=1)
        self.leaky_relu3 = nn.LeakyReLU()
        self.ConvTransBlock = [ConvTransBlock(num_filters, num_filters, 32, 4, 0, 'W' if not i%2 else 'SW')
                      for i in range(2)]
        self.TCM = nn.Sequential(*self.ConvTransBlock)
        self.conv4 = nn.Conv2d(2 * num_filters, num_filters, 3, stride=1, padding=1)
        self.leaky_relu4 = nn.LeakyReLU()
        self.conv5 = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        # print(x.shape)
        x = self.leaky_relu2(self.conv2(x))
        # print(x.shape)
        x = self.leaky_relu3(self.conv3(x))
        x = self.TCM(x)
        # print(x.shape)
        x = self.leaky_relu4(self.conv4(x))
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        return x


class Hyper_synthesis(nn.Module):
    def __init__(self, num_filters=128):
        super(Hyper_synthesis, self).__init__()
        self.conv1 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2 = nn.ConvTranspose2d(num_filters, 2 * num_filters, 3, stride=2, padding=1, output_padding=1)
        self.leaky_relu2 = nn.LeakyReLU()
        self.ConvTransBlock = [ConvTransBlock(num_filters, num_filters, 32, 4, 0, 'W' if not i % 2 else 'SW')
                               for i in range(2)]
        self.TCM = nn.Sequential(*self.ConvTransBlock)
        self.conv3 = nn.ConvTranspose2d(2 * num_filters, int(num_filters * 1.5), 3, stride=1, padding=1)
        self.leaky_relu3 = nn.LeakyReLU()
        self.conv4 = nn.ConvTranspose2d(int(num_filters * 1.5), int(num_filters * 1.5), 3, stride=2, padding=1,
                                        output_padding=1)
        self.leaky_relu4 = nn.LeakyReLU()
        # self.conv5 = nn.ConvTranspose2d(int(num_filters*1.5), num_filters*2, 3, stride=1, padding=1)
        self.conv5 = nn.ConvTranspose2d(int(num_filters * 1.5), num_filters * 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2(x))
        x = self.TCM(x)
        x = self.leaky_relu3(self.conv3(x))
        x = self.leaky_relu4(self.conv4(x))
        x = self.conv5(x)
        return x


if __name__ == "__main__":
    Hyper_analysis = Hyper_analysis()
    input_image = torch.zeros([1, 128, 16, 16])
    result = Hyper_analysis(input_image)
    print(result.shape)

    Hyper_synthesis = Hyper_synthesis()
    input_image_1 = torch.zeros([1, 128, 4, 4])
    result_1 = Hyper_synthesis(input_image_1)
    print(result_1.shape)