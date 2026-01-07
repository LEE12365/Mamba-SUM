import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from timm.models.layers import DropPath


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块，用于融合PET和CT特征"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # PET作为query，CT作为key和value
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm = nn.LayerNorm(dim)

    def forward(self, pet_feat, ct_feat):
        """PET特征作为query，CT特征作为key和value"""
        B, C, H, W = pet_feat.shape
        L = H * W

        # 将特征展平为序列
        pet_seq = rearrange(pet_feat, 'b c h w -> b (h w) c')
        ct_seq = rearrange(ct_feat, 'b c h w -> b (h w) c')

        # 生成query, key, value
        q = self.q_proj(pet_seq)  # PET作为query
        k = self.k_proj(ct_seq)  # CT作为key
        v = self.v_proj(ct_seq)  # CT作为value

        # 多头注意力
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加权融合
        x = (attn @ v)
        x = rearrange(x, 'b h l d -> b l (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)

        # 残差连接
        x = pet_seq + x
        x = self.norm(x)

        # 恢复空间维度
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x


class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(dim // reduction, 4)

        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim * 3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, low, mid, high):
        """融合低、中、高频特征"""
        B, C, _, _ = low.shape

        # 计算注意力权重
        cat_feat = torch.cat([low, mid, high], dim=1)
        pooled = self.avg_pool(cat_feat).view(B, -1)
        weights = self.mlp(pooled).view(B, 3, C, 1, 1)

        # 加权融合
        low_weight, mid_weight, high_weight = weights[:, 0], weights[:, 1], weights[:, 2]
        fused = low * low_weight + mid * mid_weight + high * high_weight

        return fused


class EfficientBlock(nn.Module):
    """高效的特征提取块"""

    def __init__(self, dim, expand_ratio=2):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)

        self.conv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False),
        )

        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        return x + self.conv(self.norm(x))


class DownSampleBlock(nn.Module):
    """下采样模块，包含特征提取和交叉注意力融合"""

    def __init__(self, in_dim, out_dim, has_cross_attn=True):
        super().__init__()
        self.has_cross_attn = has_cross_attn

        # 下采样
        self.down_conv = nn.Conv2d(in_dim, out_dim, 3, 2, 1)
        self.norm = nn.GroupNorm(1, out_dim)
        self.act = nn.GELU()

        # 特征提取
        self.extract_block = EfficientBlock(out_dim)

        # 交叉注意力融合
        if has_cross_attn:
            self.cross_fusion = CrossAttentionFusion(out_dim, num_heads=8)

    def forward(self, pet_feat, ct_feat=None):
        # 下采样
        pet_down = self.act(self.norm(self.down_conv(pet_feat)))
        pet_down = self.extract_block(pet_down)

        # 如果有CT特征且需要交叉注意力
        if self.has_cross_attn and ct_feat is not None:
            # CT也进行下采样
            ct_down = self.act(self.norm(self.down_conv(ct_feat)))
            # 交叉注意力融合
            pet_down = self.cross_fusion(pet_down, ct_down)

        return pet_down


class UpSampleBlock(nn.Module):
    """上采样模块"""

    def __init__(self, in_dim, out_dim, skip_dim=0):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        )

        if skip_dim > 0:
            self.skip_conv = nn.Conv2d(skip_dim, out_dim, 1)

        self.norm = nn.GroupNorm(1, out_dim)
        self.act = nn.GELU()
        self.extract_block = EfficientBlock(out_dim)

    def forward(self, x, skip=None):
        x = self.up_conv(x)

        if skip is not None:
            skip = self.skip_conv(skip)
            x = x + skip

        x = self.act(self.norm(x))
        x = self.extract_block(x)

        return x


class WaveletTransform(nn.Module):
    """小波变换模块"""

    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x):
        """4通道小波分解"""
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        ll = x1 + x2 + x3 + x4
        hl = -x1 - x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hh = x1 - x2 - x3 + x4

        return ll, hl, lh, hh


class FrequencyFusionBlock(nn.Module):
    """频域融合模块"""

    def __init__(self, dim):
        super().__init__()
        self.wavelet = WaveletTransform()

        # 低频处理
        self.low_conv = nn.Conv2d(dim * 3, dim, 3, 1, 1)
        self.low_block = EfficientBlock(dim)

        # 高频处理
        self.high_fusion = MultiScaleFusion(dim)
        self.high_block = EfficientBlock(dim)

    def forward(self, pet_feat, pet_down, ct_feat):
        """融合PET和CT的频域特征"""
        # 小波分解
        pet_ll, pet_hl, pet_lh, pet_hh = self.wavelet(pet_feat)
        ct_ll, ct_hl, ct_lh, ct_hh = self.wavelet(ct_feat)

        # 低频融合
        low_cat = torch.cat([pet_ll, pet_down, ct_ll], dim=1)
        low_fused = self.low_conv(low_cat)
        low_fused = self.low_block(low_fused)

        # 高频融合
        pet_high = torch.stack([pet_hl, pet_lh, pet_hh], dim=1)
        ct_high = torch.stack([ct_hl, ct_lh, ct_hh], dim=1)

        # 多尺度高频融合
        high_fused = self.high_fusion(pet_hl, pet_lh, pet_hh)
        high_fused = high_fused + self.high_fusion(ct_hl, ct_lh, ct_hh)
        high_fused = self.high_block(high_fused)

        return low_fused, high_fused


class DualBranchEncoder(nn.Module):
    """双分支编码器"""

    def __init__(self, in_ch=3, base_dim=48):
        super().__init__()

        # PET分支初始卷积
        self.pet_init = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, 3, 1, 1),
            nn.GroupNorm(1, base_dim),
            nn.GELU()
        )

        # CT分支初始卷积
        self.ct_init = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, 3, 1, 1),
            nn.GroupNorm(1, base_dim),
            nn.GELU()
        )

        # 下采样阶段
        self.down1 = DownSampleBlock(base_dim, base_dim * 2)
        self.down2 = DownSampleBlock(base_dim * 2, base_dim * 4)
        self.down3 = DownSampleBlock(base_dim * 4, base_dim * 8)

        # 频域融合模块
        self.freq_fusion1 = FrequencyFusionBlock(base_dim)
        self.freq_fusion2 = FrequencyFusionBlock(base_dim * 2)
        self.freq_fusion3 = FrequencyFusionBlock(base_dim * 4)

    def forward(self, pet_img, ct_img):
        # 初始特征提取
        pet_feat = self.pet_init(pet_img)
        ct_feat = self.ct_init(ct_img)

        # 第一层
        pet_down1 = self.down1(pet_feat, ct_feat)
        low1, high1 = self.freq_fusion1(pet_feat, pet_down1, ct_feat)

        # 第二层
        pet_down2 = self.down2(pet_down1, F.avg_pool2d(ct_feat, 2))
        low2, high2 = self.freq_fusion2(pet_down1, pet_down2, F.avg_pool2d(ct_feat, 2))

        # 第三层
        pet_down3 = self.down3(pet_down2, F.avg_pool2d(ct_feat, 4))
        low3, high3 = self.freq_fusion3(pet_down2, pet_down3, F.avg_pool2d(ct_feat, 4))

        features = {
            'low': [low1, low2, low3],
            'high': [high1, high2, high3],
            'bottleneck': pet_down3
        }

        return features


class Decoder(nn.Module):
    """解码器"""

    def __init__(self, base_dim=48):
        super().__init__()

        # 上采样阶段
        self.up3 = UpSampleBlock(base_dim * 8, base_dim * 4, skip_dim=base_dim * 4)
        self.up2 = UpSampleBlock(base_dim * 4, base_dim * 2, skip_dim=base_dim * 2)
        self.up1 = UpSampleBlock(base_dim * 2, base_dim, skip_dim=base_dim)

        # 高频特征融合
        self.high_fusion3 = MultiScaleFusion(base_dim * 4)
        self.high_fusion2 = MultiScaleFusion(base_dim * 2)
        self.high_fusion1 = MultiScaleFusion(base_dim)

        # 最终重建
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, 3, 1, 1),
            nn.GroupNorm(1, base_dim),
            nn.GELU(),
            nn.Conv2d(base_dim, 3, 3, 1, 1)
        )

    def forward(self, features):
        low_feats = features['low']
        high_feats = features['high']
        bottleneck = features['bottleneck']

        # 第三层上采样
        up3 = self.up3(bottleneck, low_feats[2])
        high_fused3 = self.high_fusion3(*high_feats)
        up3 = up3 + high_fused3

        # 第二层上采样
        up2 = self.up2(up3, low_feats[1])
        high_fused2 = self.high_fusion2(high_feats[0], high_feats[1], high_feats[2])
        up2 = up2 + high_fused2

        # 第一层上采样
        up1 = self.up1(up2, low_feats[0])
        high_fused1 = self.high_fusion1(*high_feats[:2])
        up1 = up1 + high_fused1

        # 最终重建
        output = self.final_conv(up1)

        return output


class WaveFusionNet(nn.Module):
    """主网络：基于小波变换和交叉注意力的PET-CT融合网络"""

    def __init__(self, in_ch=3, base_dim=48):
        super().__init__()

        # 双分支编码器
        self.encoder = DualBranchEncoder(in_ch, base_dim)

        # 解码器
        self.decoder = Decoder(base_dim)

        # 残差连接
        self.residual_conv = nn.Conv2d(in_ch, base_dim, 3, 1, 1)

    def forward(self, pet_img, ct_img):
        """前向传播
        Args:
            pet_img: 低剂量PET图像 [B, C, H, W]
            ct_img: CT图像 [B, C, H, W]
        """
        # 编码提取特征
        features = self.encoder(pet_img, ct_img)

        # 解码重建
        decoded = self.decoder(features)

        # 残差连接
        residual = self.residual_conv(pet_img)
        decoded = decoded + residual

        # 最终输出
        output = decoded + pet_img

        return output

    def test(self, device='cpu'):
        """测试网络"""
        device = torch.device(device)
        input_pet = torch.rand(1, 3, 256, 256)
        input_ct = torch.rand(1, 3, 256, 256)

        output = self.forward(input_pet, input_ct)
        print(f"Input PET shape: {input_pet.shape}")
        print(f"Input CT shape: {input_ct.shape}")
        print(f"Output shape: {output.shape}")
        print("Test passed successfully!")

        return output