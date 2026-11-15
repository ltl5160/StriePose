import torch
import torch.nn as nn
import torch.nn.functional as F
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import timm


# ============================================================
# 1. 基础卷积模块
# ============================================================

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1):
    padding = kernel_size // 2

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# ============================================================
# 2. 轻量 FPN
# ============================================================

class LightweightFPN(nn.Module):

    def __init__(self, channels, fpn_channels=64):
        super().__init__()

        c2, c3, c4, c5 = channels

        # 横向连接：统一通道数
        self.lateral_c2 = nn.Conv2d(c2, fpn_channels, 1)
        self.lateral_c3 = nn.Conv2d(c3, fpn_channels, 1)
        self.lateral_c4 = nn.Conv2d(c4, fpn_channels, 1)
        self.lateral_c5 = nn.Conv2d(c5, fpn_channels, 1)

        # 平滑卷积
        self.smooth_p2 = conv_bn_relu(fpn_channels, fpn_channels, 3)
        self.smooth_p3 = conv_bn_relu(fpn_channels, fpn_channels, 3)
        self.smooth_p4 = conv_bn_relu(fpn_channels, fpn_channels, 3)
        self.smooth_p5 = conv_bn_relu(fpn_channels, fpn_channels, 3)

    def forward(self, features):

        c2, c3, c4, c5 = features

        l2 = self.lateral_c2(c2)
        l3 = self.lateral_c3(c3)
        l4 = self.lateral_c4(c4)
        l5 = self.lateral_c5(c5)

        # Top-down pathway
        p5 = self.smooth_p5(l5)

        p4 = self.smooth_p4(
            l4 + F.interpolate(
                p5,
                size=l4.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        )

        p3 = self.smooth_p3(
            l3 + F.interpolate(
                p4,
                size=l3.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        )

        p2 = self.smooth_p2(
            l2 + F.interpolate(
                p3,
                size=l2.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        )

        return p2, p3


# ============================================================
# 3. 精简 ASPP
# ============================================================

class LightweightASPP(nn.Module):

    def __init__(self, in_channels=64, out_channels=128):
        super().__init__()

        branch_channels = 128

        # 1×1 分支
        self.branch1 = conv_bn_relu(
            in_channels, branch_channels, kernel_size=1
        )

        # 空洞卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels, branch_channels,
                kernel_size=3, padding=2, dilation=2,
                bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels, branch_channels,
                kernel_size=3, padding=4, dilation=4,
                bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(
                in_channels, branch_channels,
                kernel_size=3, padding=6, dilation=6,
                bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 全局池化分支
        # 不使用 BatchNorm，避免 batch_size=1 时 1×1 特征导致训练报错
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 5个分支拼接：5 × 128 = 640
        self.project = nn.Sequential(
            nn.Conv2d(
                branch_channels * 5,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        size = x.shape[-2:]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        b5 = self.global_pool(x)

        b5 = F.interpolate(
            b5,
            size=size,
            mode='bilinear',
            align_corners=False
        )

        out = torch.cat([b1, b2, b3, b4, b5], dim=1)

        out = self.project(out)

        return out


# ============================================================
# 4. RepViT-M1.1 + Lightweight FPN + Lightweight ASPP
# ============================================================

class RepViTNet(nn.Module):

    def __init__(
        self,
        binary_code_length=24,
        model_name='repvit_m1_1',
        pretrained=False,
        fpn_channels=64,
        aspp_channels=128
    ):
        super().__init__()

        self.num_classes = binary_code_length + 1

        # ----------------------------------------------------
        # Backbone
        # ----------------------------------------------------
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        backbone_channels = self.backbone.feature_info.channels()

        # print("RepViT feature channels:", backbone_channels)

        # ----------------------------------------------------
        # FPN
        # ----------------------------------------------------
        self.fpn = LightweightFPN(
            channels=backbone_channels,
            fpn_channels=fpn_channels
        )

        # ----------------------------------------------------
        # ASPP
        # 输入 P3：64 × 32 × 32
        # 输出：128 × 32 × 32
        # ----------------------------------------------------
        self.aspp = LightweightASPP(
            in_channels=fpn_channels,
            out_channels=aspp_channels
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        # 32×32 -> 64×64
        # ASPP输出128通道，与P2的64通道融合
        self.decoder_64 = nn.Sequential(
            conv_bn_relu(
                aspp_channels + fpn_channels,
                128,
                kernel_size=3
            ),
            conv_bn_relu(128, 128, kernel_size=3)
        )

        # 64×64 -> 128×128
        self.decoder_128 = nn.Sequential(
            conv_bn_relu(128, 64, kernel_size=3),
            conv_bn_relu(64, 64, kernel_size=3)
        )

        # 最终预测头
        self.prediction_head = nn.Conv2d(
            64,
            self.num_classes,
            kernel_size=1
        )

    def forward(self, x):

        input_size = x.shape[-2:]

        # 1. Backbone
        features = self.backbone(x)

        # 2. FPN
        p2, p3 = self.fpn(features)

        # 3. ASPP
        aspp_feature = self.aspp(p3)

        # 4. 32×32 -> 64×64
        x = F.interpolate(
            aspp_feature,
            size=p2.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # 与FPN高分辨率特征融合
        x = torch.cat([x, p2], dim=1)

        x = self.decoder_64(x)

        # 5. 64×64 -> 128×128
        output_size = (
            input_size[0] // 2,
            input_size[1] // 2
        )

        x = F.interpolate(
            x,
            size=output_size,
            mode='bilinear',
            align_corners=False
        )

        x = self.decoder_128(x)

        # 6. Mask + Binary Code
        output = self.prediction_head(x)

        mask, binary_code = torch.split(
            output,
            [1, self.num_classes - 1],
            dim=1
        )

        return mask, binary_code
    






# import timm

# import torch
# import torch.nn as nn
# import timm


# class RepViTNet(nn.Module):
#     """
#     RepViT Backbone + Original DeepLabV3 ASPP

#     Input:
#         x: [B, 3, 256, 256]

#     Output:
#         mask:        [B, 1, 128, 128]
#         binary_code: [B, binary_code_length, 128, 128]
#     """

#     def __init__(
#         self,
#         binary_code_length,
#         output_kernel_size=1,
#         model_name='repvit_m1_1',
#         pretrained=False,
#         out_index=1,
#         aspp_channels=512
#     ):
#         super(RepViTNet, self).__init__()

#         self.num_classes = binary_code_length + 1

#         # 1. RepViT Backbone
#         self.backbone = timm.create_model(
#             model_name,
#             pretrained=pretrained,
#             features_only=True,
#             out_indices=(out_index,)
#         )

#         # Get feature channels automatically
#         in_channels = self.backbone.feature_info.channels()[0]

#         # 2. Channel projection
#         self.proj = nn.Sequential(
#             nn.Conv2d(
#                 in_channels,
#                 aspp_channels,
#                 kernel_size=1,
#                 bias=False
#             ),
#             nn.BatchNorm2d(aspp_channels),
#             nn.ReLU(inplace=True)
#         )

#         # 3. Original ASPP
#         self.aspp = ASPP(
#             num_classes=self.num_classes,
#             concat=False,
#             output_kernel_size=output_kernel_size
#         )

#     def forward(self, x):

#         # RepViT feature extraction
#         features = self.backbone(x)
#         feature_map = features[0]
#         print("Backbone:", feature_map.shape)

#         # Channel projection
#         feature_map = self.proj(feature_map)
#         print("After projection:", feature_map.shape)

#         # ASPP prediction
#         output = self.aspp(feature_map)
#         print("ASPP output:", output.shape)

#         # Split Mask and Binary Code
#         mask, binary_code = torch.split(
#             output,
#             [1, self.num_classes - 1],
#             dim=1
#         )

#         return mask, binary_code