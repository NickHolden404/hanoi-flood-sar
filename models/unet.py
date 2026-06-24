"""
models/unet.py

Baseline U-Net for binary SAR flood segmentation.
Follows Ronneberger et al. (2015) with standard modifications:
  - Configurable encoder depth via encoder_channels list
  - BatchNorm2d after every conv
  - Dropout2d after first conv in each encoder block
  - ConvTranspose2d upsampling (no bilinear)
  - Single-channel raw logit output (no sigmoid — applied in loss / metrics)

Architecture (default encoder_channels=[16,32,64,128,256]):
  Encoder: 4 down-blocks  bottleneck at 256ch
  Decoder: 4 up-blocks with skip concatenation
  Output : Conv2d(16, 1, 1×1)
"""
import torch
import torch.nn as nn


# ── building blocks ────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two 3×3 conv layers with BN + ReLU. Dropout on first conv."""

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


class UpBlock(nn.Module):
    """ConvTranspose2d upsample  concatenate skip  ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ── U-Net ──────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Args:
        in_channels     : number of input channels (7 for full stack,
                          fewer for ablation configs)
        encoder_channels: list of channel widths for encoder blocks + bottleneck
                          e.g. [16, 32, 64, 128, 256]
                          len(encoder_channels) - 1 = number of down-blocks
        dropout         : Dropout2d probability in ConvBlocks
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
        for ch_out in encoder_channels[:-1]:          # all but bottleneck
            self.enc_blocks.append(ConvBlock(ch_in, ch_out, dropout))
            ch_in = ch_out

        # ── bottleneck ────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(ch_in, encoder_channels[-1], dropout)

        # ── decoder ───────────────────────────────────────────────────────
        # decoder_channels mirrors encoder in reverse
        dec_chs = list(reversed(encoder_channels))    # [256,128,64,32,16]
        self.up_blocks = nn.ModuleList()
        for i in range(len(dec_chs) - 1):
            in_ch   = dec_chs[i]
            skip_ch = dec_chs[i + 1]
            out_ch  = dec_chs[i + 1]
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch, dropout))

        # ── output head ───────────────────────────────────────────────────
        self.head = nn.Conv2d(encoder_channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder — save skip connections
        skips = []
        for block in self.enc_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        return self.head(x)          # (B, 1, H, W) raw logit


# ── factory ────────────────────────────────────────────────────────────────

def get_unet(cfg: dict, n_channels: int = None) -> UNet:
    """Build U-Net from config dict."""
    mc  = cfg["unet"]
    nch = n_channels if n_channels is not None else cfg["dataset"]["n_channels"]
    return UNet(
        in_channels      = nch,
        encoder_channels = mc["encoder_channels"],
        dropout          = mc["dropout"],
    )