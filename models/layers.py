import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def upconv(in_c, out_c):
    return nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)


class ChannelAttention(nn.Module):
    def __init__(self, in_c, r=8):
        super().__init__()
        mid = max(1, in_c // r)
        self.fc1 = nn.Conv2d(in_c, mid, 1, bias=False)
        self.fc2 = nn.Conv2d(mid, in_c, 1, bias=False)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # [B,1,H,W]
        return x * a, a


class CrossGate(nn.Module):
    """
    Cross-gating fusion: gate each stream with the other's attention map.
    eye_fused  = eye * sigmoid(Conv([attn_eye, attn_gland]))
    gland_fused= gland * sigmoid(Conv([attn_gland, attn_eye]))
    """
    def __init__(self, in_c):
        super().__init__()
        self.gate_eye = nn.Conv2d(2, 1, 1, bias=True)
        self.gate_gland = nn.Conv2d(2, 1, 1, bias=True)

    def forward(self, eye_feat, gland_feat, a_eye, a_gland):
        if a_eye is None:
            a_eye = torch.ones_like(eye_feat[:, :1])
        if a_gland is None:
            a_gland = torch.ones_like(gland_feat[:, :1])

        g_e = torch.sigmoid(self.gate_eye(torch.cat([a_eye, a_gland], dim=1)))
        g_g = torch.sigmoid(self.gate_gland(torch.cat([a_gland, a_eye], dim=1)))

        return eye_feat * g_e, gland_feat * g_g


class DualAttention(nn.Module):
    """Selectively applies channel+spatial attention per branch."""
    def __init__(self, in_c, use_channel=True, k=7, mode="both"):
        super().__init__()
        assert mode in {"none", "eye-only", "gland-only", "both"}
        self.mode = mode
        self.use_channel = use_channel

        self.ca_eye = (
            ChannelAttention(in_c)
            if (use_channel and mode in {"eye-only", "both"})
            else nn.Identity()
        )
        self.ca_gland = (
            ChannelAttention(in_c)
            if (use_channel and mode in {"gland-only", "both"})
            else nn.Identity()
        )

        self.sa_eye = SpatialAttention(k=k) if mode in {"eye-only", "both"} else None
        self.sa_gland = SpatialAttention(k=k) if mode in {"gland-only", "both"} else None

    def forward(self, x):
        # eye branch
        if self.mode in {"eye-only", "both"}:
            xe = self.ca_eye(x)
            xe, a_eye = self.sa_eye(xe)
        else:
            xe, a_eye = x, None

        # gland branch
        if self.mode in {"gland-only", "both"}:
            xg = self.ca_gland(x)
            xg, a_gland = self.sa_gland(xg)
        else:
            xg, a_gland = x, None

        if self.mode == "none":
            xe, xg = x, x

        return xe, xg, a_eye, a_gland
