"""
models/attention_unet.py

Attention U-Net for binary SAR flood segmentation.
Follows Oktay et al. (2018) — additive soft attention gates on every
skip connection.

Attention gate (additive, Oktay et al. 2018):
    g  : gating signal from decoder (coarser resolution, upsampled)
    x  : skip connection from encoder (finer resolution)
    α  = σ( ψ( ReLU( W_g(g) + W_x(x) ) ) )   ∈ (0,1) per spatial location
    output = α ⊙ x

Design rationale for SAR urban flood mapping:
    Attention gates learn to suppress activations in irrelevant spatial
    regions (building shadow zones with near-zero backscatter) by using
    the decoder's current semantic context (gating signal g) to weight
    encoder skip features spatially. This provides direct inductive bias
    against the principal false-positive error class in urban SAR imagery.

Architecture identical to U-Net baseline except skip connections are
attention-gated. All other components (ConvBlock, UpBlock structure,
loss, optimizer) are unchanged for fair comparison.

Output: single-channel raw logit (B, 1, H, W) — no sigmoid.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── shared building blocks ─────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two 3×3 conv + BN + ReLU. Dropout2d on first conv."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ── attention gate ─────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Additive attention gate (Oktay et al. 2018).

    Args:
        F_g   : channels in gating signal g (from decoder)
        F_l   : channels in skip connection x (from encoder)
        F_int : intermediate channels (typically F_l // 2)
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        # Project gating signal to F_int
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        # Project skip features to F_int
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        # Collapse F_int  1 spatial attention map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g : gating signal  (B, F_g, H/2, W/2) — from decoder, coarser
        x : skip features  (B, F_l, H,   W  ) — from encoder, finer
        Returns attended skip: α ⊙ x, same shape as x.
        """
        # Upsample g to match x spatial size
        g_up = F.interpolate(self.W_g(g), size=x.shape[2:],
                             mode="bilinear", align_corners=False)
        wx   = self.W_x(x)
        α    = self.psi(self.relu(g_up + wx))   # (B, 1, H, W)
        return α * x                             # broadcast over channels


# ── attention up-block ─────────────────────────────────────────────────────

class AttentionUpBlock(nn.Module):
    """
    ConvTranspose2d upsample  AttentionGate on skip  concatenate  ConvBlock.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.attn = AttentionGate(
            F_g=in_ch // 2,
            F_l=skip_ch,
            F_int=skip_ch // 2,
        )
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = self.up(x)
        skip = self.attn(g=x, x=skip)       # gate the skip with upsampled x
        x    = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ── Attention U-Net ────────────────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Args:
        in_channels     : input channels (7 for full stack, fewer for ablation)
        encoder_channels: channel widths — same convention as UNet baseline
                          e.g. [16, 32, 64, 128, 256]
        dropout         : Dropout2d probability
    """

    def __init__(self, in_channels: int = 7,
                 encoder_channels: list = None,
                 dropout: float = 0.3):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [16, 32, 64, 128, 256]

        # ── encoder ───────────────────────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        self.pool       = nn.MaxPool2d(2, 2)

        ch_in = in_channels
        for ch_out in encoder_channels[:-1]:
            self.enc_blocks.append(ConvBlock(ch_in, ch_out, dropout))
            ch_in = ch_out

        # ── bottleneck ────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(ch_in, encoder_channels[-1], dropout)

        # ── decoder with attention gates ──────────────────────────────────
        dec_chs = list(reversed(encoder_channels))   # [256,128,64,32,16]
        self.up_blocks = nn.ModuleList()
        for i in range(len(dec_chs) - 1):
            in_ch   = dec_chs[i]
            skip_ch = dec_chs[i + 1]
            out_ch  = dec_chs[i + 1]
            self.up_blocks.append(
                AttentionUpBlock(in_ch, skip_ch, out_ch, dropout))

        # ── output head ───────────────────────────────────────────────────
        self.head = nn.Conv2d(encoder_channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips = []
        for block in self.enc_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder — attention-gated skips
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        return self.head(x)    # (B, 1, H, W) raw logit


# ── factory ────────────────────────────────────────────────────────────────

def get_attention_unet(cfg: dict, n_channels: int = None) -> AttentionUNet:
    """Build AttentionUNet from config dict."""
    mc  = cfg["attention_unet"]
    nch = n_channels if n_channels is not None else cfg["dataset"]["n_channels"]
    return AttentionUNet(
        in_channels      = nch,
        encoder_channels = mc["encoder_channels"],
        dropout          = mc["dropout"],
    )