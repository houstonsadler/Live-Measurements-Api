import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # you already have timm in requirements.txt

class MidasSmall(nn.Module):
    """
    Rough equivalent of MiDaS_small / MiDaS v2.1 small.
    It uses an EfficientNet-Lite backbone (tf_efficientnet_lite3) and a lightweight decoder.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone (feature extractor)
        # This pulls EfficientNet-Lite3 from timm.
        self.backbone = timm.create_model(
            "tf_efficientnet_lite3",
            features_only=True,
            pretrained=False
        )
        # features_only=True makes it return intermediate feature maps

        # The decoder expects feature maps from multiple scales.
        # We'll build a simple upsample/merge decoder head.

        # Figure out output channels of each stage from the backbone
        channels = self.backbone.feature_info.channels()  # e.g. [24, 32, 48, 136, 384] etc.
        self.channels = channels

        # We'll compress those to a shared channel dim for fusion
        reduce_dim = 64

        self.reduce_layers = nn.ModuleList([
            nn.Conv2d(c, reduce_dim, kernel_size=1) for c in channels
        ])

        # final fusion conv to predict depth
        self.pred = nn.Conv2d(reduce_dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B,3,H,W), 0..1 normalized

        feats = self.backbone(x)
        # feats is a list of feature maps at different resolutions, deepest last

        # reduce each to same #channels (64), then upsample all to the size of the FIRST map
        target_h, target_w = feats[0].shape[2], feats[0].shape[3]

        fused = 0
        for feat, reduce in zip(feats, self.reduce_layers):
            z = reduce(feat)
            if z.shape[2] != target_h or z.shape[3] != target_w:
                z = F.interpolate(z, size=(target_h, target_w),
                                  mode="bilinear",
                                  align_corners=False)
            fused = fused + z

        out = self.pred(fused)  # (B,1,H,W)
        # For depth we usually don't need activation here; caller can postprocess/log-scale
        return out
