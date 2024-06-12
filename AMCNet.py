
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from toolbox.backbone.pvt.pvtv2_encoder import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from toolbox.backbone.ResNet import Backbone_ResNet50_in3, Backbone_ResNet50_in1
import math
from torch.nn import functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from sklearn.cluster import KMeans

# 主干网络和融合方式不变，再次设计一个模块
class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _ConvBNSig(nn.Module):
    """Conv-BN-Sigmoid"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, **kwargs):
        super(_ConvBNSig, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class DWI(nn.Module):
    def __init__(self, channel1, channel2, outsize):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel1, channel2, kernel_size=1),
            nn.BatchNorm2d(channel2),
            nn.ReLU()
        )
        self.dept = nn.Conv2d(channel2, channel2, kernel_size=(1, 1), bias=False)
        self.rgb = nn.Conv2d(channel2, channel2, kernel_size=(1, 1), bias=False)

        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(_ConvBNReLU(channel2, 24, 1, 1), _ConvBNSig(24, outsize, 1, 1))

    def forward(self, rgb, dep):
        dep = self.conv(dep)
        assert rgb.size() == dep.size()

        rgbd = rgb + dep
        m_batchsize, C, width, height = rgb.size()

        proj_rgb = self.rgb(rgb).view(m_batchsize, -1, height * width).permute(0, 2, 1)  # B X (H*W) X C
        proj_dep = self.dept(dep).view(m_batchsize, -1, height * width)  # B X C x (H*W)
        energy = torch.bmm(proj_rgb, proj_dep) / math.sqrt(C)  # B X (H*W) X (H*W)

        attention1 = self.softmax1(energy)  # B X (H*W) X (H*W)

        att_r = torch.bmm(proj_rgb.permute(0, 2, 1), attention1)
        att_b = torch.bmm(proj_dep, attention1)
        # proj_rgbd = self.rgbd(rgbd).view(m_batchsize,-1,height*width) # B X C X (H*W)
        # attention2 = torch.bmm(proj_rgbd,attention1.permute(0,2,1) )
        attention2 = att_r + att_b
        output = attention2.view(m_batchsize, C, width, height) + rgbd

        GapOut = self.GAP(output)
        gate = self.mlp(GapOut)

        return gate

class M_F(nn.Module):
    def __init__(self):
        super(M_F, self).__init__()
        self.sa = SpatialAttention()
        self.spatial_attention1 = nn.Sequential(
            nn.Conv2d(3, 1, 3, padding=1, dilation=1),
            # nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.spatial_attention2 = nn.Sequential(
            nn.Conv2d(3, 1, 3, padding=3, dilation=3),
            # nn.ReLU(inplace=True),
            nn.Sigmoid())
        self.sp_conv = nn.Conv2d(2, 1, 1, padding=0)
        # self.ca = ChannelAttention(channel*3)

    def forward(self, fuse1,fuse2):
        SA1 = self.spatial_attention1(fuse1)
        SA2 = self.spatial_attention2(fuse2)
        SA_final = self.sp_conv(torch.cat([SA1, SA2], dim=1))

        return SA_final


class MHI(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel):
        super(MHI, self).__init__()
        self.conv1 = _ConvBNReLU(in_channel, mid_channel, kernel_size=1)
        self.conv2 = _ConvBNReLU(mid_channel, in_channel, kernel_size=1)
        self.conv3 = _ConvBNReLU(out_channel, mid_channel, kernel_size=1)
        self.M_F_r = M_F()
        self.alpha_S = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        self.ca = ChannelAttention(out_channel)
        self.r1 = nn.Conv2d(mid_channel, mid_channel//2, kernel_size=3, padding=1, dilation=1, stride=1)
        self.r3 = nn.Conv2d(mid_channel, mid_channel//2, kernel_size=3, padding=3, dilation=3, stride=1)
        self.r5 = nn.Conv2d(mid_channel, mid_channel//2, kernel_size=3, padding=5, dilation=5, stride=1)
    def forward(self, x, y, Quality_D):
        # x 表示rgb， y 表示dep
        alpha_y = Quality_D * self.conv3(y)
        fuse = self.conv1(x) * alpha_y + self.conv1(x)
        r1 = self.r1(fuse)
        r3 = self.r3(fuse)
        r5 = self.r5(fuse)
        avg_out_r1 = torch.mean(r1, dim=1, keepdim=True)
        max_out_r1, _ = torch.max(r1, dim=1, keepdim=True)

        avg_out_r3 = torch.mean(r3, dim=1, keepdim=True)
        max_out_r3, _ = torch.max(r3, dim=1, keepdim=True)

        avg_out_r5 = torch.mean(r5, dim=1, keepdim=True)
        max_out_r5, _ = torch.max(r5, dim=1, keepdim=True)

        fuse_r1 = torch.cat([avg_out_r1, avg_out_r3, avg_out_r5], dim=1)
        fuse_r3 = torch.cat([max_out_r1, max_out_r3, max_out_r5], dim=1)

        fuse_weight = self.M_F_r(fuse_r1, fuse_r3)
        fuse = fuse + fuse_weight * fuse
        # print(fuse)
        return fuse


class LCE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LCE, self).__init__()
        self.relu = nn.LeakyReLU(True)
        self.branch0 = nn.Sequential(
            _ConvBNReLU(in_channel, out_channel, kernel_size=3, padding=1),
        )
        self.branch1 = nn.Sequential(
            _ConvBNReLU(in_channel, in_channel // 4, 1),
            _ConvBNReLU(in_channel // 4, in_channel // 4, kernel_size=(1, 3), padding=(0, 1)),
            _ConvBNReLU(in_channel // 4, in_channel // 4, kernel_size=(3, 1), padding=(1, 0)),
            _ConvBNReLU(in_channel // 4, out_channel, kernel_size=3, padding=1)
        )
        self.branch2 = nn.Sequential(
            _ConvBNReLU(in_channel, in_channel // 4, 1),
            _ConvBNReLU(in_channel // 4, in_channel // 4, kernel_size=(1, 5), padding=(0, 2)),
            _ConvBNReLU(in_channel // 4, in_channel // 4, kernel_size=(5, 1), padding=(2, 0)),
            _ConvBNReLU(in_channel // 4, out_channel, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            _ConvBNReLU(in_channel, in_channel // 4, 1),
            _ConvBNReLU(in_channel // 4, in_channel // 4, kernel_size=(1, 7), padding=(0, 3)),
            _ConvBNReLU(in_channel // 4, in_channel // 4, kernel_size=(7, 1), padding=(3, 0)),
            _ConvBNReLU(in_channel // 4, out_channel, kernel_size=3, padding=1)
        )
        self.conv_cat = _ConvBNReLU(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = _ConvBNReLU(in_channel, out_channel, 1)




    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))

        return x

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder5 = nn.Sequential(
            # BasicConv2d(2048, 512, 3, padding=1),
            _ConvBNReLU(512, 512, 3, padding=1),
            _ConvBNReLU(512, 512, 3, padding=1),
            nn.Dropout(0.5),
        )
        self.up5 = TransBasicConv2d(512, 512, kernel_size=2, stride=2,
                                    padding=0, dilation=1, bias=False)
        self.S6 = nn.Conv2d(512, 6, 3, stride=1, padding=1)
        self.S5 = nn.Conv2d(512, 6, 3, stride=1, padding=1)

        self.decoder4 = nn.Sequential(
            _ConvBNReLU(1024, 512, 3, padding=1),
            _ConvBNReLU(512, 256, 3, padding=1),
            nn.Dropout(0.5),

        )
        self.up4 = TransBasicConv2d(256, 128, kernel_size=2, stride=2,
                                    padding=0, dilation=1, bias=False)
        self.S4 = nn.Conv2d(256, 6, 3, stride=1, padding=1)

        self.decoder3 = nn.Sequential(
            _ConvBNReLU(384, 256, 3, padding=1),
            _ConvBNReLU(256, 128, 3, padding=1),
            nn.Dropout(0.5),

        )
        self.up3 = TransBasicConv2d(128, 64, kernel_size=2, stride=2,
                                    padding=0, dilation=1, bias=False)
        self.S3 = nn.Conv2d(128, 6, 3, stride=1, padding=1)

        self.decoder2 = nn.Sequential(
            _ConvBNReLU(192, 128, 3, padding=1),
            _ConvBNReLU(128, 64, 3, padding=1),
            nn.Dropout(0.5),

        )
        self.up2 = TransBasicConv2d(64, 32, kernel_size=2, stride=2,
                                    padding=0, dilation=1, bias=False)
        self.S2 = nn.Conv2d(64, 6, 3, stride=1, padding=1)

        self.decoder1 = nn.Sequential(
            _ConvBNReLU(96, 32, 3, padding=1),
            # BasicConv2d(64, 32, 3, padding=1),
            # BasicConv2d(256, 128, 3, padding=1),
        )
        self.up1 = TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                                    padding=0, dilation=1, bias=False)
        self.S1 = nn.Conv2d(32, 6, 3, stride=1, padding=1)
        # self.cluster = ClusteringChannelCompression()

    def forward(self, x5, x4, x3, x2, x1):
        # x5: 1/16, 512; x4: 1/8, 512; x3: 1/4, 256; x2: 1/2, 128; x1: 1/1, 64
        # print(x5.shape)
        # 512 256 128 64    16 32 64 128
        # z6 = self.S6(x5)
        x5_up = self.decoder5(x5)
        z5 = self.S5(x5_up)

        x4_up = self.decoder4(torch.cat((x4, self.up5(x5_up)), 1))
        # print('x4_up size {} '.format(x4_up.shape))
        z4 = self.S4(x4_up)

        x3_up = self.decoder3(torch.cat((x3, self.up4(x4_up)), 1))
        # print('x3_up size {} '.format(x3_up.shape))
        z3 = self.S3(x3_up)
        # print(z3.shape)
        x2_up = self.decoder2(torch.cat((x2, self.up3(x3_up)), 1))
        # print('x2_up size {} '.format(x2_up.shape))
        z2 = self.S2(x2_up)
        # print(z2.shape)
        x1_up = self.decoder1(torch.cat((x1, self.up2(x2_up)), 1))
        z1 = self.S1(x1_up)
        # print('x1_up size {} '.format(x1_up.shape))
        s1 = self.upsample(x1_up)
        z0 = self.S1(s1)

        # print('s1 size {} '.format(s1.shape))

        return z0, z1, z2, z3, z4, z5


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=groups),
            norm_layer(out_channels),
            # nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )



class E_FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=5, act_layer=nn.ReLU6, drop=0.):
        super(E_FFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNReLU(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.conv1 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=ksize,
                                groups=hidden_features)
        self.conv2 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3,
                                groups=hidden_features)
        self.fc2 = ConvBN(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.fc2(x1 + x2)
        x = self.act(x)

        return x

class MutilScal(nn.Module):
    def __init__(self, dim=512, fc_ratio=4, dilation=[3, 5, 7], pool_ratio=16):
        super(MutilScal, self).__init__()
        self.conv0_1 = nn.Conv2d(dim, dim // fc_ratio, 1)
        self.bn0_1 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv0_2 = nn.Conv2d(dim // fc_ratio, dim // fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3],
                                 groups=dim // fc_ratio)
        self.bn0_2 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv0_3 = nn.Conv2d(dim // fc_ratio, dim, 1)
        self.bn0_3 = nn.BatchNorm2d(dim)

        self.conv1_2 = nn.Conv2d(dim // fc_ratio, dim // fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2],
                                 groups=dim // fc_ratio)
        self.bn1_2 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv1_3 = nn.Conv2d(dim // fc_ratio, dim, 1)
        self.bn1_3 = nn.BatchNorm2d(dim)

        self.conv2_2 = nn.Conv2d(dim // fc_ratio, dim // fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1],
                                 groups=dim // fc_ratio)
        self.bn2_2 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv2_3 = nn.Conv2d(dim // fc_ratio, dim, 1)
        self.bn2_3 = nn.BatchNorm2d(dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6()

        self.Avg = nn.AdaptiveAvgPool2d(pool_ratio)


    def forward(self, x):
        u = x.clone()

        attn0_1 = self.relu(self.bn0_1(self.conv0_1(x)))
        attn0_2 = self.relu(self.bn0_2(self.conv0_2(attn0_1)))
        attn0_3 = self.relu(self.bn0_3(self.conv0_3(attn0_2)))

        attn1_2 = self.relu(self.bn1_2(self.conv1_2(attn0_1)))
        attn1_3 = self.relu(self.bn1_3(self.conv1_3(attn1_2)))

        attn2_2 = self.relu(self.bn2_2(self.conv2_2(attn0_1)))
        attn2_3 = self.relu(self.bn2_3(self.conv2_3(attn2_2)))

        attn = attn0_3 + attn1_3 + attn2_3
        attn = self.relu(self.bn3(self.conv3(attn)))
        attn = attn * u

        pool = self.Avg(attn)

        return pool


class Mutilscal_MHSA(nn.Module):
    def __init__(self, dim, num_heads, atten_drop=0., proj_drop=0., dilation=[3, 5, 7], fc_ratio=4, pool_ratio=16):
        super(Mutilscal_MHSA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.MSC = MutilScal(dim=dim, fc_ratio=fc_ratio, dilation=dilation, pool_ratio=pool_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim // fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.kv = Conv(dim, 2 * dim, 1)

    def forward(self, x):
        u = x.clone()
        # print(u)
        B, C, H, W = x.shape
        kv = self.MSC(x)
        kv = self.kv(kv)

        B1, C1, H1, W1 = kv.shape

        q = rearrange(x, 'b (h d) (hh) (ww) -> (b) h (hh ww) d', h=self.num_heads,
                      d=C // self.num_heads, hh=H, ww=W)
        k, v = rearrange(kv, 'b (kv h d) (hh) (ww) -> kv (b) h (hh ww) d', h=self.num_heads,
                         d=C // self.num_heads, hh=H1, ww=W1, kv=2)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.atten_drop(attn)
        attn = attn @ v

        attn = rearrange(attn, '(b) h (hh ww) d -> b (h d) (hh) (ww)', h=self.num_heads,
                         d=C // self.num_heads, hh=H, ww=W)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u


        return attn + c_attn


class GCE(nn.Module):
    def __init__(self, dim=512, num_heads=16, mlp_ratio=4, pool_ratio=16, drop=0., dilation=[3, 5, 7],
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Mutilscal_MHSA(dim, num_heads=num_heads, atten_drop=drop, proj_drop=drop, dilation=dilation,
                                   pool_ratio=pool_ratio, fc_ratio=mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim // mlp_ratio)

        self.mlp = E_FFN(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                         drop=drop)

    def forward(self, x):
        x = self.attn(x)
        x = x + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.mlp(x))

        return x


# rgb 使用resnet dep使用PVT
class AMCNet(nn.Module):
    def __init__(self, arc='B0'):
        super(AMCNet, self).__init__()
        if arc == 'Res_50_PVT':
            print('-->using Res_50_PVT right now')
            (
                self.encoder1,
                self.encoder2,
                self.encoder4,
                self.encoder8,
                self.encoder16,
            ) = Backbone_ResNet50_in3()
            self.net2 = pvt_v2_b1()

            self.lwa = DWI(64, 256, 4)
            # self.dqw = DQW()
            self.F1 = MHI(256, 64, 128)
            self.F2 = MHI(512, 128, 256)
            self.F3 = MHI(1024, 320, 512)
            self.F4 = MHI(2048, 512, 1024)

            self.conv5 = _ConvBNReLU(1024, 512, kernel_size=3, padding=1,stride=1)
            # self.conv4 = _ConvBNReLU(1024, 512, kernel_size=3, padding=1,stride=1)

            # # Decoder 2
            self.rfb1_2 = LCE(256, 256)
            self.rfb5_2 = LCE(128, 128)
            self.rfb = LCE(64, 64)
            # [1, 3, 5], [3, 5, 7], [5, 7, 9], [7, 9, 11]

            self.b4 = GCE(dim=1024, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=[1, 3, 5])
            self.b3 = GCE(dim=512, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=[3, 5, 7])

            self.lateral_conv1 = _ConvBNReLU(128, 256, 3, stride=1, padding=1)
            self.lateral_conv2 = _ConvBNReLU(256, 512, 3, stride=1, padding=1)
            self.lateral_conv3 = _ConvBNReLU(512, 1024, 3, stride=1, padding=1)

            self.decoder = decoder()

        else:
            raise Exception("Invalid Architecture Symbol: {}".format(arc))
    def kmeans_clustering(self,feature_map, num_clusters=6, num_iterations=20):
        # 将特征图展平为二维数组
        feature_map_flat = feature_map.view(feature_map.size(0)*feature_map.size(1), -1)
        # feature_map_flat = feature_map.view(feature_map.size(0), -1)

        # 初始化聚类中心
        centroids = feature_map_flat[:num_clusters].clone()

        for _ in range(num_iterations):
            # 计算每个像素点到聚类中心的距离
            distances = torch.cdist(feature_map_flat, centroids)

            # 根据距离将像素点分配到最近的聚类中心
            _, assignments = torch.min(distances, dim=1)

            # 更新聚类中心为每个聚类的平均值
            for i in range(num_clusters):
                cluster_points = feature_map_flat[assignments == i]
                if cluster_points.size(0) > 0:
                    centroids[i] = cluster_points.mean(dim=0)

        # 根据聚类结果重构特征图
        clustered_feature_map = centroids[assignments].view(feature_map.size())
        # print(clustered_feature_map)
        return clustered_feature_map

    def forward(self, rgb, dep):
        dep1, dep2, dep3, dep4 = self.net2(dep)

        x0 = self.encoder1(rgb)
        x1 = self.encoder2(x0)
        # 设置lwa权值
        alpha_weight = self.DWI(x1, dep1)

        r1 = self.F1(x1, dep1, alpha_weight[:, 0:1, ...])
        # r1 = self.lateral_conv1(r1)
        # r1 = self.F1(x1, dep1, 1)
        x2 = self.encoder4(self.lateral_conv1(r1))
        r2 = self.F2(x2, dep2, alpha_weight[:, 1:2, ...])
        # r2 = self.lateral_conv2(r2)
        # r2 = self.F2(x2, dep2, 1)
        x3 = self.encoder8(self.lateral_conv2(r2))
        r3 = self.F3(x3, dep3, alpha_weight[:, 2:3, ...])
        # r3 = self.lateral_conv3(r3)
        # r3 = self.F3(x3, dep3, 1)
        x4 = self.encoder16(self.lateral_conv3(r3))
        r4 = self.F4(x4, dep4, alpha_weight[:, 3:4, ...])

        M4 = self.b4(r4)
        M4 = self.conv5(M4)
        H4 = self.kmeans_clustering(M4)
        M3 = self.b3(r3)
        # M3 = self.conv4(M3)
        H3 = self.kmeans_clustering(M3)
        x2_1 = self.rfb1_2(r2)  # 512->256
        x1_1 = self.rfb5_2(r1)  # 256->128
        x0 = self.rfb(x0)  # 64->64
        z0, z1, z2, z3, z4, z5 = self.decoder(M4, M3, x2_1, x1_1, x0)

        return z0, z1, z2, z3, z4, z5, H3, H4
        # return z0, z1, z2, z3, z4, z5, x2_up


if __name__ == '__main__':
    net = AMCNet('Res_50_PVT')
    rgb = torch.randn(6, 3, 256, 256)
    dep = torch.randn(6, 3, 256, 256)
    out = net(rgb, dep)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
    print(out[4].shape)
    print(out[5].shape)
    print(out[6].shape)
    print(out[7].shape)