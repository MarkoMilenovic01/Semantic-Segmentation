# phase3_model.py
# NOS – Semantic Segmentation (U-Net variant)
# Input:  (B, 3, 256, 256)
# Output: (B, 1, 256, 256)
# Convs keep size via padding=1, Up with ConvTranspose2d(k=2, s=2)

import torch
import torch.nn as nn

def _log(name, x, verbose):
    if verbose:
        print(f"{name:>16}: {tuple(x.shape)}")

class DoubleConv(nn.Module):
    """
    2x [Conv3x3 -> BN -> ReLU], keeps HxW via padding=1
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    """
    Encoder block:
      x_skip = DoubleConv(in_ch, out_ch)   # same HxW
      x_down = MaxPool2d(2)                # /2
      returns (x_down, x_skip)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, verbose=False, tag=""):
        x_skip = self.conv(x)
        _log(f"{tag} 2xConv", x_skip, verbose)
        x_down = self.pool(x_skip)
        _log(f"{tag} MaxPool", x_down, verbose)
        return x_down, x_skip

class Up(nn.Module):
    """
    Decoder block:
      x = ConvTranspose2d(in_ch, out_ch, k=2, s=2)    # *2
      x = concat(x_skip, x) along C  -> C = out_ch + skip_ch (skip_ch==out_ch here)
      x = DoubleConv(2*out_ch, out_ch)   # keeps HxW
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch=out_ch * 2, out_ch=out_ch)

    def forward(self, x, x_skip, verbose=False, tag=""):
        x = self.up(x)
        _log(f"{tag} UpSample", x, verbose)
        # shapes must match – with k=2,s=2 they do for powers of two
        x = torch.cat([x_skip, x], dim=1)
        _log(f"{tag} Concat", x, verbose)
        x = self.conv(x)
        _log(f"{tag} 2xConv", x, verbose)
        return x

class UNet(nn.Module):
    """
    Modified U-Net per assignment:
      Down(3->32) -> Down(32->64) -> Down(64->128)
      Bottleneck  128->256 (2xConv)
      Up(256->128) -> Up(128->64) -> Up(64->32)
      Final 32->1 (1x1)
    """
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

        # Encoder
        self.down1 = Down(3, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)

        # Bottleneck (no pooling)
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)

        # Final 1x1 conv -> 1 channel (logits)
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        _log("Input", x, self.verbose)

        # Encoder
        x1_down, x1_skip = self.down1(x, verbose=self.verbose, tag="Down1")
        # x1_skip: (B,32,256,256), x1_down: (B,32,128,128)

        x2_down, x2_skip = self.down2(x1_down, verbose=self.verbose, tag="Down2")
        # x2_skip: (B,64,128,128), x2_down: (B,64,64,64)

        x3_down, x3_skip = self.down3(x2_down, verbose=self.verbose, tag="Down3")
        # x3_skip: (B,128,64,64), x3_down: (B,128,32,32)

        # Bottleneck
        xb = self.bottleneck(x3_down)
        _log("Bottleneck 2xConv", xb, self.verbose)  # (B,256,32,32)

        # Decoder
        x = self.up1(xb, x3_skip, verbose=self.verbose, tag="Up1")  # -> (B,128,64,64)
        x = self.up2(x,  x2_skip, verbose=self.verbose, tag="Up2")  # -> (B,64,128,128)
        x = self.up3(x,  x1_skip, verbose=self.verbose, tag="Up3")  # -> (B,32,256,256)

        out = self.final(x)  # logits
        _log("Final Conv(1x1)", out, self.verbose)  # (B,1,256,256)
        return out

# --------------------------------------------------------------------------- #
# Standalone check: run to see sizes after every operation
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    model = UNet(verbose=True)
    x = torch.randn(1, 3, 256, 256)  # batch=1 demo
    y = model(x)
    print("\nOutput logits shape:", tuple(y.shape))
    # Tip: use BCEWithLogitsLoss during training, and torch.sigmoid(y) for masks at inference.
