import torch
import torch.nn as nn

from .layers import conv_block, upconv, DualAttention, CrossGate


class DRAMNetUniversal(nn.Module):
    def __init__(
        self,
        in_ch=1,
        out_ch=2,
        width=32,
        mode="both",
        fusion="none",
        use_channel_attn=True,
        attn_kernel=7,
    ):
        super().__init__()
        assert mode in {"none", "eye-only", "gland-only", "both"}
        assert fusion in {"none", "add", "concat", "crossgate"}

        self.mode = mode
        self.fusion = fusion

        C1, C2, C3, C4 = width, width * 2, width * 4, width * 8

        # encoder
        self.enc1 = conv_block(in_ch, C1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(C1, C2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(C2, C3)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(C3, C4)

        # dual attention at bottleneck
        self.dual_attn = DualAttention(
            C4,
            use_channel=use_channel_attn,
            k=attn_kernel,
            mode=mode,
        )

        # decoders expect C4 for all fusion modes
        dec_in = C4

        # eyelid decoder
        self.up_eye1 = upconv(dec_in, C3)
        self.dec_eye1 = conv_block(C3 + C3, C3)
        self.up_eye2 = upconv(C3, C2)
        self.dec_eye2 = conv_block(C2 + C2, C2)
        self.up_eye3 = upconv(C2, C1)
        self.dec_eye3 = conv_block(C1 + C1, C1)
        self.out_eye = nn.Conv2d(C1, 1, 1)

        # gland decoder
        self.up_g1 = upconv(dec_in, C3)
        self.dec_g1 = conv_block(C3 + C3, C3)
        self.up_g2 = upconv(C3, C2)
        self.dec_g2 = conv_block(C2 + C2, C2)
        self.up_g3 = upconv(C2, C1)
        self.dec_g3 = conv_block(C1 + C1, C1)
        self.out_g = nn.Conv2d(C1, 1, 1)

        # fusion helpers
        self.crossgate = CrossGate(C4) if fusion == "crossgate" else None
        self.reduce_cat_eye = nn.Conv2d(C4 * 2, C4, 1) if fusion == "concat" else None
        self.reduce_cat_gland = nn.Conv2d(C4 * 2, C4, 1) if fusion == "concat" else None

        # for visualization / debugging
        self.last_attn_eye = None
        self.last_attn_gland = None

    def _fuse(self, eye_b, gland_b, a_eye, a_gland):
        if self.fusion == "none":
            return eye_b, gland_b

        if self.fusion == "add":
            f = eye_b + gland_b
            return f, f

        if self.fusion == "concat":
            fe = torch.cat([eye_b, gland_b], dim=1)   # [B, 2*C4, H, W]
            fg = torch.cat([gland_b, eye_b], dim=1)   # [B, 2*C4, H, W]
            return self.reduce_cat_eye(fe), self.reduce_cat_gland(fg)

        if self.fusion == "crossgate":
            return self.crossgate(eye_b, gland_b, a_eye, a_gland)

        raise ValueError(f"Unknown fusion: {self.fusion}")

    def _decode_eye(self, feat, e1, e2, e3):
        d1 = self.up_eye1(feat)
        d1 = self.dec_eye1(torch.cat([d1, e3], dim=1))
        d2 = self.up_eye2(d1)
        d2 = self.dec_eye2(torch.cat([d2, e2], dim=1))
        d3 = self.up_eye3(d2)
        d3 = self.dec_eye3(torch.cat([d3, e1], dim=1))
        return self.out_eye(d3)

    def _decode_gland(self, feat, e1, e2, e3):
        d1 = self.up_g1(feat)
        d1 = self.dec_g1(torch.cat([d1, e3], dim=1))
        d2 = self.up_g2(d1)
        d2 = self.dec_g2(torch.cat([d2, e2], dim=1))
        d3 = self.up_g3(d2)
        d3 = self.dec_g3(torch.cat([d3, e1], dim=1))
        return self.out_g(d3)

    def forward(self, x):
        # encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        # dual attention at bottleneck
        eye_b, gland_b, a_eye, a_gland = self.dual_attn(b)
        self.last_attn_eye = a_eye
        self.last_attn_gland = a_gland

        # fusion at bottleneck
        eye_b, gland_b = self._fuse(eye_b, gland_b, a_eye, a_gland)

        # decode branches
        out_eye = self._decode_eye(eye_b, e1, e2, e3)
        out_gland = self._decode_gland(gland_b, e1, e2, e3)

        # [B, 2, H, W] (eye, gland)
        return torch.cat([out_eye, out_gland], dim=1)


